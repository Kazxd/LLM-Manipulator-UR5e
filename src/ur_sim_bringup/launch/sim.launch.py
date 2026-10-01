import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    world = os.path.join(get_package_share_directory("ur_sim_bringup"), "worlds", "pick_place.sdf")

    # NOTE: verify the world argument name with:
    #   ros2 launch ur_simulation_gz ur_sim_moveit.launch.py --show-args | grep -i world
    ur_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("ur_simulation_gz"), "launch", "ur_sim_moveit.launch.py")),
        launch_arguments={"ur_type": "ur5e", "world_file": world}.items(),
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

    return LaunchDescription([ur_sim, cam_bridge, pose_bridge, pose_gt_bridge, cam_tf])
