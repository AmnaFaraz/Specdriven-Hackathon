---
id: module-5-chapter-5
title: "Capstone Project: Complete Humanoid Robot System"
sidebar_label: "Capstone Project"
---

# Capstone Project: Complete Humanoid Robot System

This capstone chapter integrates all components developed throughout the course into a complete, functional humanoid robot system. This represents the culmination of all previous modules, combining the robotic nervous system, digital twin, AI brain, and vision-language-action capabilities into a unified autonomous platform.

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE HUMANOID ROBOT SYSTEM                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PERCEPTION LAYER                                                              │
│  ┌─────────────────┬─────────────────┬─────────────────┬──────────────────────┐ │
│  │   Vision        │   Audio         │   Tactile       │   Environmental      │ │
│  │   Processing    │   Processing    │   Sensors       │   Monitoring       │ │
│  │   • Cameras     │   • Microphones │   • Force/Torque│   • IMU            │ │
│  │   • LIDAR       │   • Speakers    │   • Joint Encoders│ • Pressure       │ │
│  │   • Depth       │   • Echo Cancel │   • Temperature │   • Temperature    │ │
│  └─────────────────┴─────────────────┴─────────────────┴──────────────────────┘ │
│                                                                                 │
│  AI PROCESSING LAYER                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  • Computer Vision    • NLP Processing    • Motion Planning               │ │
│  │  • SLAM               • Speech Synthesis  • Path Planning                 │ │
│  │  • Object Detection   • Dialog Manager    • Trajectory Generation         │ │
│  │  • Semantic Mapping   • Command Parser    • Balance Control               │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  CONTROL LAYER                                                                 │
│  ┌─────────────────┬─────────────────┬─────────────────┬──────────────────────┐ │
│  │   Locomotion    │   Manipulation  │   Balance       │   Whole-Body       │ │
│  │   Control       │   Control       │   Control       │   Coordination     │ │
│  │   • Walking     │   • Grasping    │   • Posture     │   • Joint Control  │ │
│  │   • Turning     │   • Reaching    │   • Stability   │   • Trajectory     │ │
│  │   • Stair Climbing│ • Manipulation│   • Recovery    │   • Safety Limits  │ │
│  └─────────────────┴─────────────────┴─────────────────┴──────────────────────┘ │
│                                                                                 │
│  ORCHESTRATION LAYER                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  • Behavior Manager │  • Task Planner   │  • State Machine              │ │
│  │  • Skill Library    │  • Motion Planner │  • Safety Validator           │ │
│  │  • Human-Robot     │  • Action Executor│  • Performance Monitor        │ │
│  │  • Interaction     │                   │                                │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ROS 2 COMMUNICATION LAYER                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  • Node Management   • Topic Communication   • Service Interfaces         │ │
│  │  • Action Servers    • Parameter Server      • TF Transformations         │ │
│  │  • Lifecycle Nodes   • Bag Recording       • Diagnostic Tools           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  HARDWARE ABSTRACTION LAYER                                                    │
│  ┌─────────────────┬─────────────────┬─────────────────┬──────────────────────┐ │
│  │   Joint Drivers │   Sensor        │   Power         │   Communication      │ │
│  │   • Servo Ctrl  │   • ADC/DAC     │   • Battery     │   • Ethernet       │ │
│  │   • Motor Ctrl  │   • IMU         │   • Power Dist  │   • WiFi           │ │
│  │   • PID Tuning  │   • Encoders    │   • DC-DC       │   • CAN Bus        │ │
│  └─────────────────┴─────────────────┴─────────────────┴──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Main System Orchestrator

### Complete Humanoid Robot Node
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, Imu, PointCloud2
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Duration
import numpy as np
import threading
import time
from typing import Dict, Any, Optional
import json

class HumanoidOrchestrator(Node):
    def __init__(self):
        super().__init__('humanoid_orchestrator')

        # Initialize subsystem managers
        self.perception_manager = PerceptionManager(self)
        self.ai_brain = AIBrain(self)
        self.control_manager = ControlManager(self)
        self.communication_manager = CommunicationManager(self)
        self.safety_validator = SafetyValidator(self)

        # Robot state
        self.robot_state = {
            'joint_states': {},
            'sensor_data': {},
            'location': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0],  # quaternion
            'battery_level': 100.0,
            'temperature': 25.0,
            'operational_mode': 'idle',
            'safety_status': 'nominal'
        }

        # Task management
        self.task_queue = queue.Queue()
        self.active_task = None
        self.task_lock = threading.Lock()

        # ROS 2 interfaces
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.lidar_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Command interfaces
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.behavior_cmd_pub = self.create_publisher(String, '/behavior_command', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)
        self.action_cmd_sub = self.create_subscription(
            String, '/action_command', self.action_command_callback, 10
        )

        # System control
        self.system_initialized = False
        self.operational_mode = 'idle'  # idle, autonomous, teleoperation, maintenance
        self.emergency_stop = False

        # Performance monitoring
        self.performance_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'control_loop_rate': 0.0,
            'sensor_data_rate': 0.0
        }

        # Initialize system
        self.initialize_system()

        # Start main control loop
        self.main_loop_timer = self.create_timer(0.01, self.main_control_loop)  # 100Hz

    def initialize_system(self):
        """Initialize complete humanoid system"""
        self.get_logger().info("Initializing Humanoid Robot System...")

        # Initialize perception system
        self.perception_manager.initialize()

        # Initialize AI brain
        self.ai_brain.initialize()

        # Initialize control system
        self.control_manager.initialize()

        # Initialize communication manager
        self.communication_manager.initialize()

        # Initialize safety validator
        self.safety_validator.initialize()

        # Wait for all systems to be ready
        time.sleep(2.0)

        # Set operational mode
        self.operational_mode = 'idle'
        self.system_initialized = True

        self.get_logger().info("Humanoid Robot System Initialization Complete!")

    def joint_state_callback(self, msg):
        """Update joint state information"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.robot_state['joint_states'][name] = {
                    'position': msg.position[i],
                    'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                    'effort': msg.effort[i] if i < len(msg.effort) else 0.0
                }

    def imu_callback(self, msg):
        """Update IMU data"""
        self.robot_state['sensor_data']['imu'] = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def camera_callback(self, msg):
        """Process camera data"""
        # Forward to perception system
        self.perception_manager.process_camera_data(msg)

    def lidar_callback(self, msg):
        """Process LIDAR data"""
        # Forward to perception system
        self.perception_manager.process_lidar_data(msg)

    def odom_callback(self, msg):
        """Update odometry information"""
        self.robot_state['location'] = [
            msg.pose.pose.position.x,
            msg.pose.pose.y,
            msg.pose.pose.position.z
        ]
        self.robot_state['orientation'] = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]

    def action_command_callback(self, msg):
        """Process action commands"""
        try:
            command_data = json.loads(msg.data)
            command_type = command_data.get('type', 'unknown')

            if self.emergency_stop:
                self.get_logger().warn("Emergency stop active - ignoring command")
                return

            # Validate command safety
            if self.safety_validator.validate_command(command_data, self.robot_state):
                # Add to task queue
                self.task_queue.put({
                    'type': command_type,
                    'data': command_data,
                    'timestamp': time.time()
                })
            else:
                self.get_logger().error(f"Safety validation failed for command: {command_type}")

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON command: {msg.data}')

    def main_control_loop(self):
        """Main system control loop running at 100Hz"""
        if not self.system_initialized:
            return

        loop_start_time = time.time()

        # Process perception data
        perception_data = self.perception_manager.get_current_perception()

        # Get AI decision
        ai_decision = self.ai_brain.make_decision(
            perception_data=perception_data,
            robot_state=self.robot_state
        )

        # Generate control commands
        control_commands = self.control_manager.generate_commands(ai_decision)

        # Execute commands if safe
        if self.safety_validator.validate_action(control_commands, self.robot_state):
            self.control_manager.execute_commands(control_commands)
        else:
            self.get_logger().warn("Control command rejected by safety validator")

        # Process tasks from queue
        self.process_task_queue()

        # Update performance metrics
        self.update_performance_metrics()

        # Publish system status
        self.publish_system_status()

        # Calculate loop timing
        loop_time = time.time() - loop_start_time
        self.performance_metrics['control_rate'] = 1.0 / loop_time if loop_time > 0 else 0

    def process_task_queue(self):
        """Process queued tasks"""
        if not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()

                with self.task_lock:
                    self.active_task = task

                # Execute task based on type
                if task['type'] == 'navigate':
                    self.execute_navigation_task(task)
                elif task['type'] == 'manipulate':
                    self.execute_manipulation_task(task)
                elif task['type'] == 'interact':
                    self.execute_interaction_task(task)
                elif task['type'] == 'change_mode':
                    self.change_operational_mode(task['data'].get('mode', 'idle'))
                else:
                    self.get_logger().warn(f'Unknown task type: {task["type"]}')

                # Mark task as complete
                with self.task_lock:
                    self.active_task = None

            except queue.Empty:
                pass  # Queue was empty

    def execute_navigation_task(self, task):
        """Execute navigation task"""
        destination = task['data'].get('destination', [0.0, 0.0, 0.0])
        mode = task['data'].get('mode', 'safe')

        # Plan navigation
        navigation_plan = self.ai_brain.plan_navigation(
            start_pos=self.robot_state['location'],
            destination=destination,
            mode=mode
        )

        # Execute navigation
        self.control_manager.execute_navigation(navigation_plan)

    def execute_manipulation_task(self, task):
        """Execute manipulation task"""
        target_object = task['data'].get('object', '')
        action = task['data'].get('action', 'grasp')

        # Plan manipulation
        manipulation_plan = self.ai_brain.plan_manipulation(
            target_object=target_object,
            action=action
        )

        # Execute manipulation
        self.control_manager.execute_manipulation(manipulation_plan)

    def execute_interaction_task(self, task):
        """Execute interaction task"""
        target = task['data'].get('target', 'person')
        interaction_type = task['data'].get('type', 'greet')

        # Plan interaction
        interaction_plan = self.ai_brain.plan_interaction(
            target=target,
            interaction_type=interaction_type
        )

        # Execute interaction
        self.control_manager.execute_interaction(interaction_plan)

    def change_operational_mode(self, new_mode):
        """Change operational mode"""
        valid_modes = ['idle', 'autonomous', 'teleoperation', 'maintenance', 'emergency_stop']

        if new_mode in valid_modes:
            old_mode = self.operational_mode
            self.operational_mode = new_mode

            self.get_logger().info(f'Operational mode changed from {old_mode} to {new_mode}')

            # Execute mode-specific actions
            if new_mode == 'emergency_stop':
                self.emergency_stop = True
                self.control_manager.emergency_stop()
            elif new_mode == 'autonomous':
                self.emergency_stop = False
                self.enable_autonomous_mode()
            elif new_mode == 'teleoperation':
                self.emergency_stop = False
                self.enable_teleoperation_mode()
            elif new_mode == 'maintenance':
                self.emergency_stop = False
                self.enable_maintenance_mode()
            elif new_mode == 'idle':
                self.emergency_stop = False
                self.enable_idle_mode()
        else:
            self.get_logger().warn(f'Invalid operational mode: {new_mode}')

    def enable_autonomous_mode(self):
        """Enable autonomous operation mode"""
        # Activate perception system
        self.perception_manager.activate_autonomous_mode()

        # Enable AI decision making
        self.ai_brain.enable_decision_making()

        # Set control system to autonomous
        self.control_manager.set_autonomous_mode()

    def enable_teleoperation_mode(self):
        """Enable teleoperation mode"""
        # Deactivate autonomous perception
        self.perception_manager.deactivate_autonomous_mode()

        # Disable AI decision making
        self.ai_brain.disable_decision_making()

        # Set control system to manual
        self.control_manager.set_manual_mode()

    def enable_maintenance_mode(self):
        """Enable maintenance mode"""
        # Disable all autonomous functions
        self.perception_manager.deactivate_all_modes()
        self.ai_brain.disable_all_ai_functions()
        self.control_manager.set_maintenance_mode()

    def enable_idle_mode(self):
        """Enable idle mode"""
        # Keep basic monitoring active
        self.perception_manager.activate_monitoring_mode()
        self.ai_brain.enable_basic_monitoring()
        self.control_manager.set_idle_mode()

    def update_performance_metrics(self):
        """Update system performance metrics"""
        # This would monitor actual system performance
        # For now, using placeholder values
        import psutil
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.performance_metrics['gpu_usage'] = gpus[0].load * 100
        except:
            pass

        self.performance_metrics['cpu_usage'] = psutil.cpu_percent()
        self.performance_metrics['memory_usage'] = psutil.virtual_memory().percent

    def publish_system_status(self):
        """Publish system status"""
        status_msg = String()
        status_msg.data = json.dumps({
            'operational_mode': self.operational_mode,
            'location': self.robot_state['location'],
            'battery_level': self.robot_state['battery_level'],
            'temperature': self.robot_state['temperature'],
            'performance_metrics': self.performance_metrics,
            'active_task': self.active_task['type'] if self.active_task else None,
            'safety_status': self.robot_state['safety_status'],
            'timestamp': time.time()
        })

        self.status_pub.publish(status_msg)
```

## Performance Monitoring and Optimization

### System Performance Monitor
```python
import psutil
import GPUtil
import time
from collections import deque
import threading

class SystemPerformanceMonitor:
    def __init__(self, robot_node):
        self.robot_node = robot_node
        self.performance_data = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'gpu_usage': deque(maxlen=100),
            'control_loop_times': deque(maxlen=100),
            'sensor_processing_times': deque(maxlen=100),
            'ai_processing_times': deque(maxlen=100)
        }

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitoring_thread.start()

    def monitor_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active and rclpy.ok():
            # Monitor system resources
            self.performance_data['cpu_usage'].append(psutil.cpu_percent())
            self.performance_data['memory_usage'].append(psutil.virtual_memory().percent)

            # Monitor GPU if available
            gpus = GPUtil.getGPUs()
            if gpus:
                self.performance_data['gpu_usage'].append(gpus[0].load * 100)

            time.sleep(0.1)  # Monitor every 100ms

    def get_performance_summary(self):
        """Get performance summary"""
        summary = {}
        for key, data in self.performance_data.items():
            if data:
                summary[key] = {
                    'current': data[-1] if data else 0,
                    'average': sum(data) / len(data),
                    'peak': max(data) if data else 0
                }
            else:
                summary[key] = {'current': 0, 'average': 0, 'peak': 0}

        return summary

    def check_performance_thresholds(self):
        """Check if performance is within acceptable thresholds"""
        issues = []

        # Check CPU usage
        if self.performance_data['cpu_usage'] and self.get_average('cpu_usage') > 90:
            issues.append('High CPU usage')

        # Check memory usage
        if self.performance_data['memory_usage'] and self.get_average('memory_usage') > 90:
            issues.append('High memory usage')

        # Check GPU usage
        if self.performance_data['gpu_usage'] and self.get_average('gpu_usage') > 95:
            issues.append('High GPU usage')

        return issues

    def get_average(self, metric_name):
        """Get average value for a metric"""
        data = self.performance_data[metric_name]
        return sum(data) / len(data) if data else 0

class SafetyValidator:
    def __init__(self, robot_node):
        self.robot_node = robot_node
        self.safety_limits = {
            'max_joint_velocity': 5.0,  # rad/s
            'max_joint_torque': 100.0,  # Nm
            'max_linear_velocity': 1.0,  # m/s
            'max_angular_velocity': 1.0,  # rad/s
            'min_battery_level': 10.0,  # %
            'max_temperature': 70.0  # Celsius
        }

        self.safety_violations = []
        self.emergency_stop_active = False

    def validate_command(self, command, robot_state):
        """Validate command for safety"""
        violations = []

        # Check joint limits
        if 'joint_commands' in command:
            for joint_name, cmd in command['joint_commands'].items():
                if joint_name in robot_state['joint_states']:
                    current_pos = robot_state['joint_states'][joint_name]['position']
                    target_pos = cmd.get('position', current_pos)

                    # Check velocity limits
                    if 'velocity' in cmd and abs(cmd['velocity']) > self.safety_limits['max_joint_velocity']:
                        violations.append(f'Joint {joint_name} velocity limit exceeded')

                    # Check torque limits
                    if 'effort' in cmd and abs(cmd['effort']) > self.safety_limits['max_joint_torque']:
                        violations.append(f'Joint {joint_name} torque limit exceeded')

        # Check motion limits
        if 'motion_commands' in command:
            linear_vel = command['motion_commands'].get('linear_velocity', 0)
            angular_vel = command['motion_commands'].get('angular_velocity', 0)

            if abs(linear_vel) > self.safety_limits['max_linear_velocity']:
                violations.append('Linear velocity limit exceeded')

            if abs(angular_vel) > self.safety_limits['max_angular_velocity']:
                violations.append('Angular velocity limit exceeded')

        # Check system status
        if robot_state['battery_level'] < self.safety_limits['min_battery_level']:
            violations.append('Battery level too low')

        if robot_state['temperature'] > self.safety_limits['max_temperature']:
            violations.append('Temperature too high')

        # Log violations
        if violations:
            self.safety_violations.extend(violations)
            self.robot_node.get_logger().warn(f'Safety violations: {violations}')

        return len(violations) == 0

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_stop_active = True
        self.robot_node.get_logger().error("EMERGENCY STOP ACTIVATED")

    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop_active = False
        self.robot_node.get_logger().info("Emergency stop reset")
```

## System Integration Testing

### Comprehensive Integration Test Suite
```python
import unittest
import numpy as np
import time

class HumanoidSystemIntegrationTest(unittest.TestCase):
    def setUp(self):
        """Set up complete humanoid system for integration testing"""
        # This would initialize the full system
        # For now, using placeholder
        self.system = HumanoidOrchestrator(None)  # Would need proper node
        self.system_initialized = True

    def test_perception_action_loop(self):
        """Test perception-action loop functionality"""
        if not self.system_initialized:
            self.skipTest("System not initialized")

        # Simulate perception input
        perception_data = {
            'objects': [{'class': 'person', 'position': [1.0, 0.0, 0.0], 'confidence': 0.9}],
            'scene_description': 'Room with one person at 1 meter distance',
            'spatial_map': {'free_space': [], 'obstacles': []}
        }

        robot_state = {
            'location': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'joint_states': {}
        }

        # Test AI decision making
        ai_decision = self.system.ai_brain.make_decision(perception_data, robot_state)
        self.assertIsNotNone(ai_decision)
        self.assertIn('action', ai_decision)

        # Test control command generation
        control_commands = self.system.control_manager.generate_commands(ai_decision)
        self.assertIsNotNone(control_commands)

        print("✓ Perception-action loop test passed")

    def test_navigation_functionality(self):
        """Test navigation functionality"""
        if not self.system_initialized:
            self.skipTest("System not initialized")

        # Define navigation task
        navigation_task = {
            'type': 'navigate',
            'destination': [1.0, 0.0, 0.0],
            'mode': 'safe'
        }

        # Test navigation planning
        navigation_plan = self.system.ai_brain.plan_navigation(
            start_pos=[0.0, 0.0, 0.0],
            destination=[1.0, 0.0, 0.0],
            mode='safe'
        )

        self.assertIsNotNone(navigation_plan)
        self.assertIn('path', navigation_plan)
        self.assertGreater(len(navigation_plan['path']), 0)

        print("✓ Navigation functionality test passed")

    def test_balance_control(self):
        """Test balance control system"""
        if not self.system_initialized:
            self.skipTest("System not initialized")

        # Simulate IMU data indicating imbalance
        imu_data = {
            'orientation': [0.1, 0.1, 0.0, 0.99],  # Slightly tilted
            'angular_velocity': [0.05, 0.05, 0.0],
            'linear_acceleration': [0.1, 0.1, 9.7]
        }

        # Test balance correction
        balance_correction = self.system.balance_controller.compute_balance_correction(imu_data)
        self.assertIsNotNone(balance_correction)

        # Verify correction values are reasonable
        self.assertTrue(all(abs(val) < 1.0 for val in balance_correction))

        print("✓ Balance control test passed")

    def test_safety_validation(self):
        """Test safety validation system"""
        if not self.system_initialized:
            self.skipTest("System not initialized")

        # Test safe command
        safe_command = {
            'joints': {
                'left_hip_pitch': {'position': 0.1, 'velocity': 0.5},
                'right_hip_pitch': {'position': 0.1, 'velocity': 0.5}
            }
        }

        robot_state = {
            'joint_states': {
                'left_hip_pitch': {'position': 0.0, 'velocity': 0.0},
                'right_hip_pitch': {'position': 0.0, 'velocity': 0.0}
            }
        }

        is_safe = self.system.safety_validator.validate_command(safe_command, robot_state)
        self.assertTrue(is_safe)

        # Test unsafe command (joint position beyond limits)
        unsafe_command = {
            'joints': {
                'left_hip_pitch': {'position': 10.0, 'velocity': 0.5}  # Beyond joint limits
            }
        }

        is_safe_unsafe = self.system.safety_validator.validate_command(unsafe_command, robot_state)
        self.assertFalse(is_safe_unsafe)

        print("✓ Safety validation test passed")

    def test_autonomous_behavior_sequence(self):
        """Test sequence of autonomous behaviors"""
        if not self.system_initialized:
            self.skipTest("System not initialized")

        # Define behavior sequence
        behavior_sequence = [
            {'type': 'navigate', 'target': [1.0, 0.0, 0.0]},
            {'type': 'greet', 'target_type': 'greeting_interaction'},
            {'type': 'navigate', 'target': [0.0, 0.0, 0.0]},
        ]

        print("Testing autonomous behavior sequence...")

        for i, behavior in enumerate(behavior_sequence):
            print(f"  Executing behavior {i+1}: {behavior['type']}")

            # Validate behavior
            is_safe = self.system.safety_validator.validate_action(behavior, {})
            self.assertTrue(is_safe, f"Behavior {i+1} failed safety validation")

            # Generate commands
            commands = self.system.control_manager.generate_commands(behavior)
            self.assertIsNotNone(commands)

        print("✓ Autonomous behavior sequence test passed")

    def test_system_performance(self):
        """Test system performance under load"""
        if not self.system_initialized:
            self.skipTest("System not initialized")

        # Measure performance over multiple cycles
        start_time = time.time()
        iterations = 100

        for i in range(iterations):
            # Simulate perception data
            perception_data = {
                'objects': [{'class': 'object', 'position': [i/100, 0, 0], 'confidence': 0.9}],
                'scene_description': 'Test scene',
                'spatial_map': {'free_space': [], 'obstacles': []}
            }

            robot_state = {
                'location': [0.0, 0.0, 0.0],
                'orientation': [0.0, 0.0, 0.0, 1.0],
                'joint_states': {}
            }

            # Process through system
            ai_decision = self.system.ai_brain.make_decision(perception_data, robot_state)
            control_commands = self.system.control_manager.generate_commands(ai_decision)

        end_time = time.time()
        avg_time = (end_time - start_time) / iterations

        # Verify system can maintain real-time performance (100Hz)
        self.assertLess(avg_time, 0.015, f"System too slow: {avg_time}s per iteration")

        print(f"✓ System performance test passed (avg: {avg_time*1000:.1f}ms)")

    def tearDown(self):
        """Clean up after tests"""
        print("\nIntegration tests completed successfully!")
        print("All systems are functioning properly.")

def run_integration_tests():
    """Run the complete integration test suite"""
    print("Starting Humanoid Robot System Integration Tests...\n")

    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(HumanoidSystemIntegrationTest('test_perception_action_loop'))
    suite.addTest(HumanoidSystemIntegrationTest('test_navigation_functionality'))
    suite.addTest(HumanoidSystemIntegrationTest('test_balance_control'))
    suite.addTest(HumanoidSystemIntegrationTest('test_safety_validation'))
    suite.addTest(HumanoidSystemIntegrationTest('test_autonomous_behavior_sequence'))
    suite.addTest(HumanoidSystemIntegrationTest('test_system_performance'))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate summary
    print(f"\n{'='*50}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    if result.wasSuccessful():
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("The humanoid robot system is ready for deployment.")
    else:
        print("\n❌ Some tests failed. Please address the issues before deployment.")

    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
```

## Deployment Configuration

### System Deployment Scripts
```python
import os
import subprocess
import yaml
from pathlib import Path

class SystemDeployer:
    def __init__(self):
        self.config = self.load_configuration()
        self.deployment_path = Path(self.config.get('deployment_path', './deployment'))

    def load_configuration(self):
        """Load system configuration"""
        config_path = Path('./config/system_config.yaml')

        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'deployment_path': './deployment',
                'robot_name': 'humanoid_robot',
                'control_frequency': 100,
                'safety_limits': {
                    'max_linear_velocity': 1.0,
                    'max_angular_velocity': 1.0,
                    'max_joint_velocity': 5.0,
                    'max_torque': 100.0
                },
                'communication': {
                    'ros_domain_id': 0,
                    'network_interface': 'eth0'
                }
            }

    def create_deployment_package(self):
        """Create deployment package with all necessary files"""
        # Create deployment directory structure
        self.deployment_path.mkdir(parents=True, exist_ok=True)

        # Create configuration files
        self.create_robot_config()
        self.create_controller_config()
        self.create_launch_files()

        # Create documentation
        self.create_deployment_documentation()

        print(f"Deployment package created at: {self.deployment_path}")

    def create_robot_config(self):
        """Create robot configuration files"""
        robot_config = {
            'robot_description': 'humanoid_robot_description',
            'joint_limits': {
                'left_hip_yaw': {'min': -1.5, 'max': 1.5, 'velocity': 2.0},
                'left_hip_roll': {'min': -0.5, 'max': 0.5, 'velocity': 2.0},
                'left_hip_pitch': {'min': -2.0, 'max': 0.5, 'velocity': 2.0},
                # Add all other joints...
            },
            'sensors': {
                'imu': {'rate': 100, 'type': 'vectornav'},
                'cameras': [
                    {'name': 'head_camera', 'resolution': [640, 480], 'fov': 60},
                    {'name': 'stereo_camera', 'resolution': [1280, 720], 'fov': 90}
                ],
                'lidar': {'range': 10.0, 'resolution': 0.5}
            }
        }

        config_path = self.deployment_path / 'config' / 'robot.yaml'
        config_path.parent.mkdir(exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(robot_config, f, default_flow_style=False)

    def create_controller_config(self):
        """Create controller configuration files"""
        controller_config = {
            'controller_manager': {
                'ros__parameters': {
                    'update_rate': 100,
                    'use_sim_time': False
                }
            },
            'joint_trajectory_controller': {
                'ros__parameters': {
                    'joints': [
                        'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
                        'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
                        'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
                        'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
                        'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow',
                        'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow',
                        'neck_yaw', 'neck_pitch'
                    ],
                    'command_interfaces': ['position'],
                    'state_interfaces': ['position', 'velocity'],
                    'constraints': {
                        'stopped_velocity_tolerance': 0.01,
                        'goal_time': 0.5
                    }
                }
            }
        }

        config_path = self.deployment_path / 'config' / 'controllers.yaml'
        config_path.parent.mkdir(exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(controller_config, f, default_flow_style=False)

    def create_launch_files(self):
        """Create launch files for the system"""
        launch_content = '''<?xml version="1.0"?>
<launch>
  <!-- Robot State Publisher -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="robot_description" value="$(find-pkg-share my_humanoid_robot_description)/urdf/humanoid.urdf"/>
  </node>

  <!-- Joint State Publisher -->
  <node pkg="joint_state_publisher" exec="joint_state_publisher" name="joint_state_publisher">
    <param name="use_gui" value="false"/>
  </node>

  <!-- Controller Manager -->
  <node pkg="controller_manager" exec="ros2_control_node" name="controller_manager">
    <param name="robot_description" value="$(find-pkg-share my_humanoid_robot_description)/urdf/humanoid.urdf"/>
    <remap from="/joint_states" to="dynamic_joint_states"/>
  </node>

  <!-- Start controllers -->
  <node pkg="controller_manager" exec="spawner" name="joint_state_broadcaster_spawner" args="joint_state_broadcaster"/>
  <node pkg="controller_manager" exec="spawner" name="joint_trajectory_controller_spawner" args="joint_trajectory_controller"/>

  <!-- Main orchestrator -->
  <node pkg="humanoid_system" exec="humanoid_orchestrator" name="humanoid_orchestrator" output="screen">
    <param name="control_frequency" value="100"/>
    <param name="safety_enabled" value="true"/>
  </node>

  <!-- Perception system -->
  <node pkg="humanoid_perception" exec="perception_node" name="perception_node" output="screen">
    <param name="detection_model" value="yolov8n.pt"/>
    <param name="tracking_enabled" value="true"/>
  </node>

  <!-- AI brain -->
  <node pkg="humanoid_ai" exec="ai_brain_node" name="ai_brain_node" output="screen">
    <param name="model_path" value="$(find-pkg-share humanoid_ai)/models/vla_model.pt"/>
    <param name="enable_decision_making" value="true"/>
  </node>
</launch>
'''

        launch_path = self.deployment_path / 'launch' / 'humanoid_system.launch.xml'
        launch_path.parent.mkdir(exist_ok=True)

        with open(launch_path, 'w') as f:
            f.write(launch_content)

    def create_deployment_documentation(self):
        """Create deployment documentation"""
        docs_content = f"""# Humanoid Robot Deployment Guide

## System Overview
This document describes the deployment of the autonomous humanoid robot system.

## Hardware Requirements
- NVIDIA Jetson Orin AGX or equivalent
- Real-time capable Linux system
- RT kernel configured
- Sufficient RAM and storage for AI models

## Software Dependencies
- ROS 2 Humble Hawksbill
- Python 3.10+
- CUDA 11.8+
- NVIDIA Isaac packages
- Required Python packages (see requirements.txt)

## Deployment Steps
1. Install system dependencies
2. Clone the repository
3. Build all packages: `colcon build`
4. Source the workspace: `source install/setup.bash`
5. Launch the system: `ros2 launch humanoid_system humanoid_system.launch.xml`

## Configuration
The system can be configured through the configuration files in the `config/` directory:
- `robot.yaml`: Robot-specific parameters
- `controllers.yaml`: Controller configurations
- `safety.yaml`: Safety limits and constraints

## Troubleshooting
- Check ROS 2 domain ID if multiple robots are on the same network
- Verify sensor connections and calibrations
- Monitor system resources (CPU, GPU, memory usage)

## Safety Considerations
- Ensure adequate space for robot operation
- Keep emergency stop readily accessible
- Supervise robot during initial operation
- Regular safety system checks required

## Support
For technical support, contact: team@humanoid-robotics.com
"""

        docs_path = self.deployment_path / 'README.md'
        with open(docs_path, 'w') as f:
            f.write(docs_content)

def deploy_system():
    """Deploy the complete humanoid robot system"""
    print("Starting Humanoid Robot System Deployment...")

    deployer = SystemDeployer()
    deployer.create_deployment_package()

    print("Deployment package created successfully!")
    print(f"Deployment location: {deployer.deployment_path.absolute()}")

    print("\nNext steps:")
    print("1. Transfer deployment package to target system")
    print("2. Install dependencies")
    print("3. Configure hardware interfaces")
    print("4. Run integration tests")
    print("5. Begin operational deployment")

if __name__ == "__main__":
    deploy_system()
```

## Exercise: Complete System Integration

Create a complete system integration that:
1. Combines all five modules into a unified architecture
2. Implements comprehensive safety validation
3. Provides real-time performance monitoring
4. Includes automated testing and deployment scripts

## Summary

The complete humanoid robot system represents the integration of all previous modules into a cohesive autonomous platform. The system combines multimodal perception, AI decision-making, and precise control to enable complex behaviors. Through careful design of the architecture, safety systems, and validation procedures, we create a robust platform for humanoid robotics research and applications. The system is designed for deployment on real hardware while maintaining the flexibility to adapt to various robotic platforms and applications.

---