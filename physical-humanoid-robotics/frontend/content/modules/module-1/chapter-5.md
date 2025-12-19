---
id: module-1-chapter-5
title: "ROS 2 Best Practices for Humanoid Systems"
sidebar_label: "Best Practices"
---

# ROS 2 Best Practices for Humanoid Systems

This chapter covers best practices and design patterns specifically for humanoid robot systems using ROS 2.

## Real-time Considerations

Humanoid robots often require real-time performance for stability and safety. Here are key considerations:

### Real-time Setup
```bash
# Configure system for real-time performance
echo 'session required pam_limit.so rtprio 99' | sudo tee -a /etc/pam.d/common-session
echo 'session required pam_limit.so memlock unlimited' | sudo tee -a /etc/pam.d/common-session
```

### Real-time Scheduling in ROS 2
```python
import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node
import threading
import os

class RealtimeController(Node):
    def __init__(self):
        super().__init__('realtime_controller')

        # Set real-time priority for critical threads
        self.set_realtime_priority()

        # Create timer with high frequency for control loop
        self.control_timer = self.create_timer(
            0.001,  # 1kHz control loop
            self.control_callback,
            clock=self.get_clock()
        )

    def set_realtime_priority(self):
        try:
            import sched
            import os
            # Set real-time scheduling policy (requires proper system setup)
            pid = os.getpid()
            # Note: This requires proper system configuration and privileges
        except ImportError:
            self.get_logger().warn("Real-time scheduling not available")

    def control_callback(self):
        # Critical control code here
        # This should execute in under 1ms to maintain 1kHz
        pass
```

## Safety and Fault Tolerance

Safety is paramount in humanoid robotics:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import time

class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')

        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )

        self.emergency_stop_pub = self.create_publisher(
            Bool, 'emergency_stop', 10
        )

        self.joint_limits = {
            'left_hip_pitch': (-1.5, 1.5),
            'right_hip_pitch': (-1.5, 1.5),
            'left_knee': (0.0, 2.5),
            'right_knee': (0.0, 2.5),
            # Add more joint limits...
        }

        self.last_state_time = time.time()
        self.state_timeout = 0.1  # 100ms timeout

        # Timer for safety checks
        self.safety_timer = self.create_timer(0.01, self.safety_check)  # 100Hz

    def joint_state_callback(self, msg):
        self.last_state_time = time.time()

        # Check joint limits
        for i, name in enumerate(msg.name):
            if name in self.joint_limits:
                pos = msg.position[i]
                min_limit, max_limit = self.joint_limits[name]

                if pos < min_limit or pos > max_limit:
                    self.trigger_emergency_stop(f"Joint {name} out of limits: {pos}")
                    return

    def safety_check(self):
        # Check for communication timeouts
        if time.time() - self.last_state_time > self.state_timeout:
            self.trigger_emergency_stop("Joint state timeout")
            return

        # Additional safety checks can be added here
        # - Velocity limits
        # - Acceleration limits
        # - Balance checks
        # - Collision detection

    def trigger_emergency_stop(self, reason):
        self.get_logger().error(f"EMERGENCY STOP: {reason}")
        msg = Bool()
        msg.data = True
        self.emergency_stop_pub.publish(msg)
```

## Performance Optimization

Optimizing performance for humanoid robots:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from builtin_interfaces.msg import Time
import numpy as np
from scipy import linalg

class OptimizedController(Node):
    def __init__(self):
        super().__init__('optimized_controller')

        # Use appropriate QoS for performance
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.RMW_QOS_HISTORY_KEEP_LAST,
            depth=1,  # Only keep latest message
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            # Use best effort for high-frequency data where occasional drops are acceptable
        )

        self.sensor_sub = self.create_subscription(
            JointState, 'joint_states', self.optimized_callback, qos_profile
        )

        # Pre-allocate arrays to avoid memory allocation during execution
        self.joint_positions = np.zeros(18)  # Pre-allocated array
        self.joint_velocities = np.zeros(18)  # Pre-allocated array

    def optimized_callback(self, msg):
        # Convert to numpy arrays once, reuse pre-allocated arrays
        self.joint_positions[:len(msg.position)] = msg.position
        self.joint_velocities[:len(msg.velocity)] = msg.velocity

        # Perform calculations using optimized numpy operations
        # Avoid creating temporary objects in the callback
        self.compute_control_output()

    def compute_control_output(self):
        # Efficient control computation
        # Use vectorized operations where possible
        pass
```

## System Architecture Patterns

### Component-Based Architecture
```python
# Separation of concerns for humanoid robot system
class HumanoidSystemManager:
    def __init__(self):
        # Initialize subsystems
        self.motion_controller = MotionController()
        self.balance_controller = BalanceController()
        self.safety_monitor = SafetyMonitor()
        self.state_estimator = StateEstimator()

    def update(self, dt):
        # Update subsystems in appropriate order
        self.state_estimator.update(dt)
        self.balance_controller.update(dt, self.state_estimator.get_state())
        self.motion_controller.update(dt, self.balance_controller.get_commands())
        self.safety_monitor.update(dt, self.motion_controller.get_commands())
```

## Exercise: Implement a Safety Layer

Create a safety layer that monitors:
1. Joint position limits
2. Joint velocity limits
3. Communication timeouts
4. Balance thresholds
5. Emergency stop functionality

The safety layer should be able to:
- Detect dangerous conditions
- Issue emergency stops when necessary
- Provide safe recovery procedures

## Summary

This module has covered the fundamentals of ROS 2 for humanoid robot control. You now understand the core concepts, advanced communication patterns, humanoid-specific control techniques, available packages, and best practices for safety and performance. These foundations are essential for building robust humanoid robot systems.

---