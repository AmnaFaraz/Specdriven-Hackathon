---
id: module-2-chapter-2
title: "Gazebo: Physics-Based Simulation for Robotics"
sidebar_label: "Gazebo Simulation"
---

# Gazebo: Physics-Based Simulation for Robotics

Gazebo is a powerful physics-based simulation environment that provides realistic robot simulation capabilities essential for humanoid robot development.

## Overview of Gazebo

Gazebo provides:
- **Physics Simulation**: Accurate simulation of rigid body dynamics
- **Sensor Simulation**: Realistic models for cameras, LIDAR, IMU, etc.
- **Rendering**: High-quality 3D visualization
- **Plugins**: Extensible architecture for custom functionality
- **ROS Integration**: Seamless integration with ROS/ROS2

## Installing and Setting Up Gazebo

```bash
# Install Gazebo (Fortress version for ROS 2 Humble)
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins ros-humble-gazebo-dev

# Verify installation
gz sim --version
```

## Basic Gazebo World

Creating a basic world file for humanoid robot simulation:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Include the sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Your humanoid robot will be spawned here -->
  </world>
</sdf>
```

## Gazebo Plugins for Humanoid Robots

### Joint Control Plugin
```xml
<!-- In your robot's URDF/XACRO -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_humanoid</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
  </plugin>
</gazebo>
```

### Sensor Plugins
```xml
<!-- IMU sensor for balance control -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

## Launching Gazebo with ROS 2

Creating a launch file for your humanoid robot in Gazebo:

```python
# launch/humanoid_gazebo.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Get paths
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_humanoid_description = get_package_share_directory('my_humanoid_description')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py'),
        )
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_humanoid',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

## Advanced Gazebo Features for Humanoid Robots

### Contact Sensors
For foot contact detection in walking robots:

```xml
<!-- Contact sensor for foot -->
<gazebo reference="left_foot">
  <sensor name="left_foot_contact" type="contact">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <contact>
      <collision>left_foot_collision</collision>
    </contact>
    <visualize>false</visualize>
  </sensor>
</gazebo>
```

### Terrain Simulation
Creating realistic terrain for humanoid locomotion:

```xml
<!-- Uneven terrain for walking practice -->
<surface>
  <friction>
    <ode>
      <mu>1.0</mu>
      <mu2>1.0</mu2>
      <slip1>0.0</slip1>
      <slip2>0.0</slip2>
    </ode>
  </friction>
</surface>
```

## Exercise: Create a Simple Humanoid World

Create a Gazebo world file with:
1. A humanoid robot model
2. An uneven terrain
3. A simple obstacle course
4. Appropriate lighting and physics settings

## Summary

Gazebo provides a robust physics-based simulation environment that is essential for humanoid robot development. Its accurate physics simulation, sensor modeling, and ROS integration make it ideal for testing control algorithms and robot behaviors before deployment on real hardware.

---