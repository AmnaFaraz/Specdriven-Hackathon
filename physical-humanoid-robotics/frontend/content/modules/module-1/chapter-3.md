---
id: module-1-chapter-3
title: "ROS 2 for Humanoid Robot Control"
sidebar_label: "Humanoid Control"
---

# ROS 2 for Humanoid Robot Control

This chapter focuses on applying ROS 2 concepts specifically to humanoid robot control systems, which require precise coordination of multiple joints and sensors.

## Humanoid Robot Architecture

Humanoid robots present unique challenges in ROS 2 due to their complex kinematic chains and real-time control requirements. A typical humanoid robot architecture includes:

- **Joint Controllers**: Individual motor controllers for each joint
- **Sensor Managers**: IMU, force/torque sensors, cameras, etc.
- **Motion Planning**: Trajectory generation and inverse kinematics
- **Balance Control**: Center of mass management and fall prevention
- **State Estimation**: Joint position, velocity, and external forces

## Joint State Management

For humanoid robots, managing joint states is critical. The joint_state_publisher and robot_state_publisher packages are essential:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

        # Define joint names for a simple humanoid
        self.joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow'
        ]

        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

    def timer_callback(self):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts

        self.publisher_.publish(msg)
```

## Walking Pattern Generation

Humanoid robots require sophisticated walking pattern generation. ROS 2 can coordinate multiple nodes to achieve stable locomotion:

```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np

class WalkingPatternGenerator(Node):
    def __init__(self):
        super().__init__('walking_pattern_generator')
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            'joint_trajectory',
            10
        )

    def generate_step_trajectory(self, step_length, step_height, step_time):
        """Generate a trajectory for a single step"""
        msg = JointTrajectory()
        msg.joint_names = self.get_joint_names()

        # Generate trajectory points
        num_points = int(step_time * 100)  # 100Hz control
        time_step = step_time / num_points

        for i in range(num_points + 1):
            point = JointTrajectoryPoint()
            t = i * time_step / step_time  # normalized time [0, 1]

            # Generate joint positions for walking motion
            hip_pitch = self.calculate_hip_pitch(t, step_length)
            knee_angle = self.calculate_knee_angle(t, step_height)
            ankle_angle = self.calculate_ankle_angle(t)

            point.positions = self.calculate_joint_positions(
                hip_pitch, knee_angle, ankle_angle
            )
            point.time_from_start.sec = int(i * time_step)
            point.time_from_start.nanosec = int((i * time_step % 1) * 1e9)

            msg.points.append(point)

        return msg
```

## Balance Control with ROS 2

Balance control is crucial for humanoid robots. Here's how to implement a basic balance controller:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray

class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10
        )
        self.com_publisher = self.create_publisher(
            Float64MultiArray, 'center_of_mass_cmd', 10
        )

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # PID controller parameters
        self.kp = 10.0
        self.ki = 0.5
        self.kd = 2.0
        self.integral = 0.0
        self.previous_error = 0.0

    def imu_callback(self, msg):
        # Extract orientation from quaternion
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z

        # Convert to roll, pitch, yaw
        self.roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        self.pitch = math.asin(2 * (w * y - z * x))
        self.yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        # Simple balance correction
        self.balance_correction()

    def balance_correction(self):
        target_pitch = 0.0  # Target is upright
        error = target_pitch - self.pitch

        # PID control
        self.integral += error * 0.01  # Assuming 100Hz update rate
        derivative = (error - self.previous_error) / 0.01

        control_output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.previous_error = error

        # Publish correction command
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [control_output]  # Simplified for example
        self.com_publisher.publish(cmd_msg)
```

## Exercise: Implement a Simple Joint Controller

Create a ROS 2 node that implements a position controller for a single joint. The node should:
1. Subscribe to a joint command topic
2. Use a PID controller to track the desired position
3. Publish the resulting effort to an actuator interface

## Summary

ROS 2 provides the necessary tools for controlling humanoid robots, from basic joint management to complex walking and balance control. Understanding these patterns is essential for building stable and capable humanoid robots.

---