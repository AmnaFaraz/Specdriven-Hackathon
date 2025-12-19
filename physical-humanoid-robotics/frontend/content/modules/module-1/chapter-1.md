---
id: module-1-chapter-1
title: "Introduction to ROS 2: The Foundation of Robotic Communication"
sidebar_label: "ROS 2 Basics"
---

# Introduction to ROS 2: The Foundation of Robotic Communication

Welcome to Module 1: The Robotic Nervous System (ROS 2). This module will introduce you to the Robot Operating System (ROS 2), which serves as the communication backbone for modern robotics applications.

## What is ROS 2?

ROS 2 (Robot Operating System 2) is not an actual operating system, but rather a flexible framework for writing robot software. It provides a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Key Features of ROS 2:
- **Distributed Computing**: ROS 2 allows multiple processes to communicate seamlessly, whether they're running on the same machine or distributed across a network.
- **Language Independence**: Supports multiple programming languages including C++, Python, and others.
- **Real-time Support**: Enhanced real-time capabilities compared to ROS 1.
- **Security**: Built-in security features for enterprise applications.
- **DDS Integration**: Uses Data Distribution Service (DDS) for communication.

## Core Concepts

### Nodes
A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are organized into a graph structure and perform computation. ROS 2 is designed to be broken down into many nodes to make development more modular and robust.

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics and Messages
Topics are named buses over which nodes exchange messages. A node can publish messages to a topic or subscribe to messages from a topic. Messages are the data structures that are sent via topics.

### Services
Services provide a request/reply communication pattern. A service client sends a request message to a service server, which receives the request, processes it, and sends back a response message.

```python
# Service definition example
from std_msgs.msg import String

def handle_add_two_ints(request, response):
    response.sum = request.a + request.b
    return response

# Service server
service = self.create_service(AddTwoInts, 'add_two_ints', handle_add_two_ints)
```

## Exercise: Create Your First ROS 2 Node

Try creating a simple ROS 2 node that publishes "Hello, ROS 2!" to a topic every second. Use the template above as a starting point.

## Summary

ROS 2 provides the essential communication infrastructure for robotics applications. Understanding nodes, topics, services, and messages is crucial for building complex robotic systems. In the next chapter, we'll explore more advanced ROS 2 concepts and communication patterns.

---

*Continue to [Chapter 2: Advanced ROS 2 Communication Patterns](/docs/module-1-chapter-2)*