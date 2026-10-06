import os
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = get_package_share_directory("ur_sim_bringup")
    world = os.path.join(share, "worlds", "pick_place.sdf")
    ur_launch = os.path.join(get_package_share_directory("ur_simulation_gz"), "launch", "ur_sim_moveit.launch.py")

    # gripper:=robotiq (default) -> arm + Robotiq 2F-85;  gripper:=magic -> bare arm, objects teleported.
    # skill_server must use the same choice (-p gripper:=magic|robotiq, default robotiq).
    gripper = LaunchConfiguration("gripper")
    is_robotiq = IfCondition(PythonExpression(["'", gripper, "' == 'robotiq'"]))
    is_magic = IfCondition(PythonExpression(["'", gripper, "' == 'magic'"]))

    # Gazebo resolves package://robotiq_description/... through this path (parent of the package share directory)
    try:
        rq_parent = os.path.dirname(get_package_share_directory("robotiq_description"))
    except Exception:
        rq_parent = ""
    resource_path = AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", rq_parent)

    ur_magic = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ur_launch),
        launch_arguments={"ur_type": "ur5e", "world_file": world}.items(),
        condition=is_magic,
    )
    ur_robotiq = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ur_launch),
        launch_arguments={
            "ur_type": "ur5e", "world_file": world,
            "description_file": os.path.join(share, "urdf", "ur_gz_gripper.urdf.xacro"),
            "controllers_file": os.path.join(share, "config", "ur_gripper_controllers.yaml"),
            "moveit_launch_file": os.path.join(share, "launch", "ur_moveit_robotiq.launch.py"),
        }.items(),
        condition=is_robotiq,
    )

    # waits for the controller manager, then loads the gripper controller
    gripper_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager",
                   "--controller-manager-timeout", "120"],
        parameters=[{"use_sim_time": True}],
        condition=is_robotiq,
    )

    cam_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="camera_bridge",
        arguments=[
            "/overhead/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/overhead/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/overhead/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/overhead/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
        ],
        parameters=[{"use_sim_time": True}],
    )

    pose_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="set_pose_bridge",
        arguments=["/world/pick_place/set_pose@ros_gz_interfaces/srv/SetEntityPose"],
    )

    # ground-truth model poses, used only by the benchmark (gz -> ROS, one way)
    pose_gt_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="pose_gt_bridge",
        arguments=["/world/pick_place/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"],
    )

    cam_tf = Node(
        package="tf2_ros", executable="static_transform_publisher", name="camera_tf",
        arguments=["--x", "0.6", "--y", "0", "--z", "1.2", "--pitch", "1.5708",
                   "--frame-id", "base_link", "--child-frame-id", "camera_link"],
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("gripper", default_value="robotiq", description="robotiq | magic"),
        resource_path, ur_magic, ur_robotiq, gripper_spawner, cam_bridge, pose_bridge, pose_gt_bridge, cam_tf,
    ])