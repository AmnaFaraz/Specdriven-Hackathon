---
id: module-1-chapter-4
title: "ROS 2 Packages for Humanoid Robots"
sidebar_label: "Humanoid Packages"
---

# ROS 2 Packages for Humanoid Robots

This chapter covers specialized ROS 2 packages and tools specifically designed for humanoid robot development and control.

## Navigation2 for Humanoid Robots

Navigation2 is the standard navigation stack for ROS 2, but it requires special configuration for humanoid robots:

```python
# Example configuration for humanoid navigation
# navigation2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: False
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
```

## Control Frameworks

### ros2_control
The ros2_control framework provides a standardized way to interface with robot hardware:

```xml
<!-- Controller Manager Configuration -->
<controller_manager>
  <rosparam file="$(find my_humanoid_robot_control)/config/humanoid_controllers.yaml"/>
  <controller name="joint_state_broadcaster" type="joint_state_broadcaster/JointStateBroadcaster"/>
  <controller name="humanoid_controller" type="position_controllers/JointTrajectoryController">
    <param name="joints">left_hip_yaw left_hip_roll left_hip_pitch left_knee left_ankle_pitch left_ankle_roll right_hip_yaw right_hip_roll right_hip_pitch right_knee right_ankle_pitch right_ankle_roll left_shoulder_pitch left_shoulder_roll left_elbow right_shoulder_pitch right_shoulder_roll right_elbow</param>
  </controller>
</controller_manager>
```

## Hardware Interface

Creating a hardware interface for humanoid robots:

```cpp
// Example hardware interface for humanoid robot
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace my_humanoid_hardware_interface
{
class MyHumanoidHardware : public hardware_interface::SystemInterface
{
public:
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override
  {
    if (SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
      return CallbackReturn::ERROR;
    }

    // Initialize joint data structures
    for (const hardware_interface::ComponentInfo & joint : info_.joints) {
      joint_names_.push_back(joint.name);

      // Initialize state and command interfaces
      if (joint.command_interfaces.size() != 1) {
        return CallbackReturn::ERROR;
      }

      if (joint.state_interfaces.size() != 2) {  // position and velocity
        return CallbackReturn::ERROR;
      }
    }

    return CallbackReturn::SUCCESS;
  }

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override
  {
    std::vector<hardware_interface::StateInterface> state_interfaces;
    for (auto i = 0u; i < joint_names_.size(); i++) {
      state_interfaces.emplace_back(
        joint_names_[i],
        hardware_interface::HW_IF_POSITION,
        &hw_positions_[i]);
      state_interfaces.emplace_back(
        joint_names_[i],
        hardware_interface::HW_IF_VELOCITY,
        &hw_velocities_[i]);
    }

    return state_interfaces;
  }

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override
  {
    std::vector<hardware_interface::CommandInterface> command_interfaces;
    for (auto i = 0u; i < joint_names_.size(); i++) {
      command_interfaces.emplace_back(
        joint_names_[i],
        hardware_interface::HW_IF_POSITION,
        &hw_commands_[i]);
    }

    return command_interfaces;
  }

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override
  {
    return SystemInterface::on_activate(previous_state);
  }

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override
  {
    return SystemInterface::on_deactivate(previous_state);
  }

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override
  {
    // Read joint states from hardware
    for (auto i = 0u; i < joint_names_.size(); i++) {
      // hw_positions_[i] = read_position_from_hardware(i);
      // hw_velocities_[i] = read_velocity_from_hardware(i);
    }
    return hardware_interface::return_type::OK;
  }

  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override
  {
    // Send commands to hardware
    for (auto i = 0u; i < joint_names_.size(); i++) {
      // write_position_to_hardware(i, hw_commands_[i]);
    }
    return hardware_interface::return_type::OK;
  }

private:
  std::vector<std::string> joint_names_;
  std::vector<double> hw_commands_;
  std::vector<double> hw_positions_;
  std::vector<double> hw_velocities_;
};
}  // namespace my_humanoid_hardware_interface

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  my_humanoid_hardware_interface::MyHumanoidHardware, hardware_interface::SystemInterface)
```

## URDF for Humanoid Robots

The Unified Robot Description Format (URDF) is crucial for humanoid robots:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.5"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
  </link>
</robot>
```

## Exercise: Create a URDF for a Simple Humanoid

Create a complete URDF file for a simple 6-DOF humanoid robot with:
1. Base link
2. Torso
3. Head
4. Two arms (shoulder, elbow)
5. Two legs (hip, knee, ankle)

## Summary

ROS 2 provides specialized packages and frameworks for humanoid robot development, from navigation to hardware control. Understanding these packages is essential for building sophisticated humanoid robots.

---