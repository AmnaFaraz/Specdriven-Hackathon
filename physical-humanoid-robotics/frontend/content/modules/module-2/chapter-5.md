---
id: module-2-chapter-5
title: "Digital Twin Integration with ROS 2"
sidebar_label: "Digital Twin Integration"
---

# Digital Twin Integration with ROS 2

This chapter explores how to integrate digital twin systems with ROS 2, creating a seamless connection between simulation and real-world robotics applications.

## Architecture for Digital Twin Integration

The integration architecture typically follows this pattern:

```
Real Robot ←→ ROS 2 Bridge ←→ Digital Twin ←→ Analysis Tools
    ↑              ↑                ↑            ↑
Hardware I/O ←→ Message Bus ←→ Simulation ←→ Visualization
```

## ROS 2 Bridge Implementation

Creating a bridge between real robot and digital twin:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import numpy as np
import time

class DigitalTwinBridge(Node):
    def __init__(self):
        super().__init__('digital_twin_bridge')

        # Real robot interfaces
        self.real_joint_sub = self.create_subscription(
            JointState, '/real/joint_states', self.real_joint_callback, 10
        )
        self.real_imu_sub = self.create_subscription(
            Imu, '/real/imu/data', self.real_imu_callback, 10
        )
        self.real_cmd_pub = self.create_publisher(
            JointState, '/real/joint_commands', 10
        )

        # Digital twin interfaces
        self.sim_joint_pub = self.create_publisher(
            JointState, '/sim/joint_states', 10
        )
        self.sim_imu_pub = self.create_publisher(
            Imu, '/sim/imu/data', 10
        )
        self.sim_cmd_sub = self.create_subscription(
            JointState, '/sim/joint_commands', self.sim_cmd_callback, 10
        )

        # Synchronization timer
        self.bridge_timer = self.create_timer(0.01, self.synchronization_callback)  # 100Hz

        # Data buffers
        self.real_joint_state = None
        self.sim_joint_state = None
        self.synchronization_enabled = True

    def real_joint_callback(self, msg):
        """Process joint states from real robot"""
        self.real_joint_state = msg
        if self.synchronization_enabled:
            # Publish to digital twin
            self.sim_joint_pub.publish(msg)

    def real_imu_callback(self, msg):
        """Process IMU data from real robot"""
        if self.synchronization_enabled:
            # Publish to digital twin with possible noise addition
            noisy_imu = self.add_simulation_noise(msg)
            self.sim_imu_pub.publish(noisy_imu)

    def sim_cmd_callback(self, msg):
        """Process commands from digital twin and send to real robot"""
        if self.synchronization_enabled:
            # Apply safety checks before sending to real robot
            safe_cmd = self.apply_safety_filters(msg)
            self.real_cmd_pub.publish(safe_cmd)

    def add_simulation_noise(self, imu_msg):
        """Add realistic noise to IMU data for simulation"""
        noisy_imu = Imu()
        noisy_imu.header = imu_msg.header

        # Add noise to angular velocity
        noisy_imu.angular_velocity.x = imu_msg.angular_velocity.x + np.random.normal(0, 0.01)
        noisy_imu.angular_velocity.y = imu_msg.angular_velocity.y + np.random.normal(0, 0.01)
        noisy_imu.angular_velocity.z = imu_msg.angular_velocity.z + np.random.normal(0, 0.01)

        # Add noise to linear acceleration
        noisy_imu.linear_acceleration.x = imu_msg.linear_acceleration.x + np.random.normal(0, 0.05)
        noisy_imu.linear_acceleration.y = imu_msg.linear_acceleration.y + np.random.normal(0, 0.05)
        noisy_imu.linear_acceleration.z = imu_msg.linear_acceleration.z + np.random.normal(0, 0.05)

        return noisy_imu

    def apply_safety_filters(self, cmd_msg):
        """Apply safety filters to commands before sending to real robot"""
        filtered_cmd = JointState()
        filtered_cmd.header = cmd_msg.header
        filtered_cmd.name = cmd_msg.name
        filtered_cmd.position = []

        for pos in cmd_msg.position:
            # Apply position limits
            filtered_pos = max(min(pos, 3.14), -3.14)  # ±π rad limits
            filtered_cmd.position.append(filtered_pos)

        return filtered_cmd

    def synchronization_callback(self):
        """Synchronize data between real robot and digital twin"""
        # This method can implement more complex synchronization logic
        # such as time alignment, data buffering, etc.
        pass
```

## Advanced Synchronization Techniques

### Time Synchronization
```python
class TimeSynchronizer:
    def __init__(self):
        self.real_time_offset = 0.0
        self.sim_time_offset = 0.0
        self.time_sync_enabled = True

    def synchronize_timestamps(self, real_msg, sim_msg):
        """Align timestamps between real and simulated systems"""
        if self.time_sync_enabled:
            # Adjust simulation time to match real time
            sim_msg.header.stamp = real_msg.header.stamp
        return real_msg, sim_msg
```

### Data Buffering for Smooth Synchronization
```python
from collections import deque
import threading

class DataBuffer:
    def __init__(self, buffer_size=100):
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()

    def add_data(self, data):
        with self.lock:
            self.buffer.append(data)

    def get_recent_data(self, n=1):
        with self.lock:
            if len(self.buffer) >= n:
                return list(self.buffer)[-n:]
            else:
                return list(self.buffer)
```

## Visualization and Monitoring

Creating a monitoring interface for the digital twin:

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class DigitalTwinMonitor:
    def __init__(self):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.real_joint_positions = []
        self.sim_joint_positions = []
        self.errors = []

        # Setup plots
        self.setup_plots()

    def setup_plots(self):
        """Setup matplotlib plots for monitoring"""
        self.axs[0, 0].set_title('Real Robot Joint Positions')
        self.axs[0, 1].set_title('Simulated Robot Joint Positions')
        self.axs[1, 0].set_title('Position Errors')
        self.axs[1, 1].set_title('Synchronization Status')

    def update_plots(self, real_pos, sim_pos):
        """Update plots with new data"""
        self.real_joint_positions.append(real_pos)
        self.sim_joint_positions.append(sim_pos)

        # Calculate errors
        if len(real_pos) == len(sim_pos):
            error = [abs(r - s) for r, s in zip(real_pos, sim_pos)]
            self.errors.append(error)

        # Update each subplot
        self.axs[0, 0].clear()
        self.axs[0, 0].plot(real_pos)
        self.axs[0, 0].set_title('Real Robot Joint Positions')

        self.axs[0, 1].clear()
        self.axs[0, 1].plot(sim_pos)
        self.axs[0, 1].set_title('Simulated Robot Joint Positions')

        self.axs[1, 0].clear()
        if self.errors:
            self.axs[1, 0].plot(self.errors[-50:])  # Show last 50 errors
        self.axs[1, 0].set_title('Position Errors')

        self.axs[1, 1].clear()
        sync_status = "Active" if self.is_synchronized() else "Drift Detected"
        self.axs[1, 1].text(0.5, 0.5, sync_status, ha='center', va='center',
                           transform=self.axs[1, 1].transAxes, fontsize=14)
        self.axs[1, 1].set_title('Synchronization Status')

        plt.tight_layout()

    def is_synchronized(self):
        """Check if real and sim are synchronized"""
        if not self.errors:
            return True
        avg_error = np.mean(self.errors[-10:]) if len(self.errors) >= 10 else 0
        return avg_error < 0.1  # Threshold for synchronization
```

## Unity Integration with ROS 2

For Unity-based digital twins, creating a ROS 2 bridge:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class UnityROSBridge : MonoBehaviour
{
    ROSConnection ros;
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;

    // Robot model references
    public Transform[] jointTransforms;
    public string[] jointNames;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);

        // Subscribe to ROS topics
        ros.Subscribe<JointStateMsg>("/real/joint_states", JointStateCallback);

        // Start publishing simulated data
        InvokeRepeating("PublishSimData", 0.0f, 0.05f);
    }

    void JointStateCallback(JointStateMsg jointState)
    {
        // Update Unity robot model based on real robot joint states
        for (int i = 0; i < jointNames.Length; i++)
        {
            int jointIndex = System.Array.IndexOf(jointState.name, jointNames[i]);
            if (jointIndex >= 0 && jointIndex < jointState.position.Length)
            {
                UpdateJoint(jointTransforms[i], jointState.position[jointIndex]);
            }
        }
    }

    void UpdateJoint(Transform joint, float angle)
    {
        // Apply rotation to joint (adjust based on joint type)
        joint.localRotation = Quaternion.Euler(0, angle * Mathf.Rad2Deg, 0);
    }

    void PublishSimData()
    {
        // Publish simulated sensor data
        var imuMsg = new ImuMsg();
        imuMsg.header.stamp = new builtin_interfaces.msg.Time();
        // Fill in IMU data from Unity physics
        ros.Publish("/sim/imu/data", imuMsg);
    }
}
```

## Exercise: Implement a Digital Twin Bridge

Create a complete digital twin bridge that:
1. Receives joint states from a real robot
2. Updates a simulation model in real-time
3. Implements safety filters for commands
4. Provides synchronization monitoring
5. Visualizes the comparison between real and simulated states

## Summary

Digital twin integration with ROS 2 enables powerful development and testing capabilities for robotic systems. By creating robust bridges between real and simulated systems, we can leverage the benefits of both environments while maintaining safety and synchronization. This approach is particularly valuable for humanoid robots where development time and safety are critical factors.

---