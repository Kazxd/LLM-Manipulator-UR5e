import os
import yaml

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder

from ament_index_python.packages import get_package_share_directory

# Robotiq 2F-85 links (prefix is empty in ur_gz_gripper.urdf.xacro)
GRIPPER_LINKS = [
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link", "robotiq_85_right_knuckle_link",
    "robotiq_85_left_inner_knuckle_link", "robotiq_85_right_inner_knuckle_link",
    "robotiq_85_left_finger_link", "robotiq_85_right_finger_link",
    "robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link",
]
ARM_LINKS = ["forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link", "flange", "tool0"]


def gripper_acm_xml():
    """disable_collisions entries: every gripper pair, and every gripper link against the last arm links."""
    pairs = []
    for i, a in enumerate(GRIPPER_LINKS):
        for b in GRIPPER_LINKS[i + 1:]:
            pairs.append((a, b))
        for b in ARM_LINKS:
            pairs.append((a, b))
    return "\n".join(f'  <disable_collisions link1="{a}" link2="{b}" reason="Robotiq"/>' for a, b in pairs) + "\n"


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path) as file:
            return yaml.safe_load(file)
    except OSError:
        return None


def declare_arguments():
    return LaunchDescription(
        [
            DeclareLaunchArgument("launch_rviz", default_value="true", description="Launch RViz?"),
            DeclareLaunchArgument(
                "ur_type",
                description="Typo/series of used UR robot.",
                choices=["ur3", "ur5", "ur10", "ur3e", "ur5e", "ur7e", "ur10e", "ur12e", "ur16e",
                         "ur8long", "ur15", "ur18", "ur20", "ur30"],
            ),
            DeclareLaunchArgument(
                "warehouse_sqlite_path",
                default_value=os.path.expanduser("~/.ros/warehouse_ros.sqlite"),
                description="Path where the warehouse database should be stored",
            ),
            DeclareLaunchArgument("launch_servo", default_value="false", description="Launch Servo?"),
            DeclareLaunchArgument("use_sim_time", default_value="false",
                                  description="Using or not time from simulation"),
            DeclareLaunchArgument("publish_robot_description_semantic", default_value="true",
                                  description="MoveGroup publishes robot description semantic"),
        ]
    )


def generate_launch_description():
    launch_rviz = LaunchConfiguration("launch_rviz")
    ur_type = LaunchConfiguration("ur_type")
    warehouse_sqlite_path = LaunchConfiguration("warehouse_sqlite_path")
    launch_servo = LaunchConfiguration("launch_servo")
    use_sim_time = LaunchConfiguration("use_sim_time")
    publish_robot_description_semantic = LaunchConfiguration("publish_robot_description_semantic")

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description_semantic(Path("srdf") / "ur.srdf.xacro", {"name": ur_type})
        .to_moveit_configs()
    )

    # add the gripper collision pairs to the SRDF text
    srdf = moveit_config.robot_description_semantic.get("robot_description_semantic")
    if isinstance(srdf, str) and "</robot>" in srdf:
        cut = srdf.rfind("</robot>")
        moveit_config.robot_description_semantic = {
            "robot_description_semantic": srdf[:cut] + gripper_acm_xml() + srdf[cut:]}
        print("[ur_moveit_robotiq] added Robotiq collision exceptions to the SRDF")
    else:
        print("[ur_moveit_robotiq] WARNING: could not patch the SRDF (unexpected type); gripper collisions not disabled")

    warehouse_ros_config = {
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
        "warehouse_host": warehouse_sqlite_path,
    }

    ld = LaunchDescription()
    ld.add_entity(declare_arguments())

    wait_robot_description = Node(
        package="ur_robot_driver",
        executable="wait_for_robot_description",
        output="screen",
    )
    ld.add_action(wait_robot_description)

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            warehouse_ros_config,
            {
                "use_sim_time": use_sim_time,
                "publish_robot_description_semantic": publish_robot_description_semantic,
            },
        ],
    )

    servo_yaml = load_yaml("ur_moveit_config", "config/ur_servo.yaml")
    servo_params = {"moveit_servo": servo_yaml}
    servo_node = Node(
        package="moveit_servo",
        condition=IfCondition(launch_servo),
        executable="servo_node",
        parameters=[moveit_config.to_dict(), servo_params, {"use_sim_time": use_sim_time}],
        output="screen",
    )

    rviz_config_file = PathJoinSubstitution([FindPackageShare("ur_moveit_config"), "config", "moveit.rviz"])
    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            warehouse_ros_config,
            {"use_sim_time": use_sim_time},
        ],
    )

    ld.add_action(
        RegisterEventHandler(
            OnProcessExit(
                target_action=wait_robot_description,
                on_exit=[move_group_node, rviz_node, servo_node],
            )
        ),
    )

    return ld