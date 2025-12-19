---
id: module-1-chapter-2
title: "Advanced ROS 2 Communication Patterns"
sidebar_label: "ROS 2 Patterns"
---

# Advanced ROS 2 Communication Patterns

In this chapter, we'll explore advanced communication patterns in ROS 2 that are essential for building complex robotic systems.

## Action Servers and Clients

Actions are a more complex form of communication that involve sending a goal, receiving feedback during processing, and returning a result. They are ideal for long-running tasks.

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from example_interfaces.action import Fibonacci


class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            self.get_logger().info('Publishing feedback: {0}'.format(
                feedback_msg.sequence))

            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence

        self.get_logger().info('Returning result: {0}'.format(result.sequence))

        return result
```

## Parameter Server

ROS 2 provides a parameter server for runtime configuration of nodes. Parameters can be declared and used to configure behavior without recompiling.

```python
# Declaring parameters
self.declare_parameter('frequency', 10)
self.declare_parameter('robot_name', 'turtlebot')

# Getting parameter values
frequency = self.get_parameter('frequency').value
robot_name = self.get_parameter('robot_name').value
```

## Lifecycle Nodes

Lifecycle nodes provide a state machine for managing the complex startup, shutdown, and error recovery processes common in robotic systems.

```python
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn


class LifecycleTalker(LifecycleNode):

    def __init__(self):
        super().__init__('lifecycle_talker')
        self.pub = None

    def on_configure(self, state: LifecycleState):
        self.pub = self.create_publisher(String, 'lifecycle_chatter', 10)
        self.get_logger().info('Configured lifecycle talker')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState):
        self.get_logger().info('Activated lifecycle talker')
        return super().on_activate(state)

    def timer_callback(self):
        msg = String()
        msg.data = 'Lifecycle chatter'
        self.pub.publish(msg)
```

## Quality of Service (QoS) Settings

QoS profiles allow you to specify delivery guarantees for topics, which is crucial for safety-critical robotic applications.

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

# Reliable communication with durability
qos_profile = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE
)

publisher = node.create_publisher(String, 'topic', qos_profile)
```

## Exercise: Implement a Lifecycle Node

Create a lifecycle node that controls a simulated motor. The node should have the following states:
1. Unconfigured → Configured (when parameters are set)
2. Inactive → Active (when motor is ready)
3. Active → Inactive (when motor is stopped)
4. Cleanup and shutdown states

## Summary

Advanced ROS 2 communication patterns provide the tools needed for complex robotic systems. Actions for long-running tasks, parameters for configuration, lifecycle nodes for state management, and QoS settings for reliability are all essential for professional robotic applications.

---

*Continue to [Chapter 3: ROS 2 for Humanoid Robot Control](/docs/module-1-chapter-3)*