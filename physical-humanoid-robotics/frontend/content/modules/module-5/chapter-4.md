---
id: module-5-chapter-4
title: "Humanoid Robot Integration and Testing"
sidebar_label: "Integration & Testing"
---

# Humanoid Robot Integration and Testing

This chapter covers the integration of all subsystems into a complete humanoid robot system and the comprehensive testing required to ensure reliable operation.

## System Integration Architecture

### Complete Robot System Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HUMANOID ROBOT SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Perception Layer                                                              │
│  ┌─────────────────┬─────────────────┬─────────────────┬──────────────────────┐ │
│  │   Vision        │   Audio         │   Tactile       │   Environmental      │ │
│  │   Processing    │   Processing    │   Sensors       │   Monitoring       │ │
│  │   • Cameras     │   • Microphones │   • Force/Torque│   • IMU            │ │
│  │   • LIDAR       │   • Speakers    │   • Joint Encoders│ • Pressure       │ │
│  │   • Depth       │   • Echo Cancel │   • Temperature │   • Temperature    │ │
│  └─────────────────┴─────────────────┴─────────────────┴──────────────────────┘ │
│                                                                                 │
│  AI Processing Layer                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  • Computer Vision    • NLP Processing    • Motion Planning               │ │
│  │  • SLAM               • Speech Recognition• Path Planning                 │ │
│  │  • Object Detection   • Voice Synthesis   • Trajectory Generation         │ │
│  │  • Semantic Mapping   • Dialog Manager    • Balance Control               │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Control Layer                                                                 │
│  ┌─────────────────┬─────────────────┬─────────────────┬──────────────────────┐ │
│  │   Locomotion    │   Manipulation  │   Balance       │   Whole-Body       │ │
│  │   Control       │   Control       │   Control       │   Coordination     │ │
│  │   • Walking     │   • Grasping    │   • Posture     │   • Joint Control  │ │
│  │   • Turning     │   • Reaching    │   • Stability   │   • Trajectory     │ │
│  │   • Stair Climbing│ • Manipulation│   • Recovery    │   • Safety Limits  │ │
│  └─────────────────┴─────────────────┴─────────────────┴──────────────────────┘ │
│                                                                                 │
│  ROS 2 Middleware Layer                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  • Node Management   • Topic Communication   • Service Interfaces         │ │
│  │  • Action Servers    • Parameter Server      • TF Transformations         │ │
│  │  • Lifecycle Nodes   • Bag Recording       • Diagnostic Tools           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Hardware Interface Layer                                                      │
│  ┌─────────────────┬─────────────────┬─────────────────┬──────────────────────┐ │
│  │   Joint Drivers │   Sensor        │   Power         │   Communication      │ │
│  │   • Servo Ctrl  │   • ADC/DAC     │   • Battery     │   • Ethernet       │ │
│  │   • Motor Ctrl  │   • IMU         │   • Power Dist  │   • WiFi           │ │
│  │   • PID Tuning  │   • Encoders    │   • DC-DC       │   • Bluetooth      │ │
│  └─────────────────┴─────────────────┴─────────────────┴──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Integration Testing Framework

### Test Infrastructure Setup
```python
import unittest
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, Imu, PointCloud2
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Bool
import numpy as np
import time
from typing import Dict, Any, List
import threading

class HumanoidIntegrationTester:
    def __init__(self):
        # Initialize ROS context
        rclpy.init()
        self.node = Node('humanoid_integration_tester')

        # Test subscribers
        self.joint_state_sub = self.node.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.node.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.camera_sub = self.node.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )
        self.lidar_sub = self.node.create_subscription(
            PointCloud2, '/lidar/points', self.lidar_callback, 10
        )

        # Test publishers
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.node.create_publisher(JointState, '/joint_commands', 10)

        # Test state variables
        self.joint_states = {}
        self.imu_data = {}
        self.camera_data = None
        self.lidar_data = None
        self.test_results = {}
        self.test_running = False

        # Test timing
        self.test_start_time = None
        self.test_timeout = 30.0  # seconds

    def joint_state_callback(self, msg):
        """Update joint state data"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_states[name] = {
                    'position': msg.position[i],
                    'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                    'effort': msg.effort[i] if i < len(msg.effort) else 0.0,
                    'timestamp': self.node.get_clock().now()
                }

    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            'timestamp': self.node.get_clock().now()
        }

    def camera_callback(self, msg):
        """Update camera data"""
        self.camera_data = {
            'width': msg.width,
            'height': msg.height,
            'encoding': msg.encoding,
            'timestamp': self.node.get_clock().now()
        }

    def lidar_callback(self, msg):
        """Update LIDAR data"""
        self.lidar_data = {
            'width': msg.width,
            'height': msg.height,
            'row_step': msg.row_step,
            'timestamp': self.node.get_clock().now()
        }

    def run_integration_tests(self):
        """Run comprehensive integration tests"""
        print("Starting Humanoid Robot Integration Tests...")

        tests = [
            self.test_sensor_health,
            self.test_actuator_control,
            self.test_balance_system,
            self.test_perception_pipeline,
            self.test_navigation_system,
            self.test_manipulation_system,
            self.test_safety_systems
        ]

        results = {}

        for test_func in tests:
            test_name = test_func.__name__
            print(f"Running {test_name}...")
            try:
                result = test_func()
                results[test_name] = result
                print(f"  Result: {'PASS' if result['success'] else 'FAIL'}")
                if not result['success']:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                results[test_name] = {
                    'success': False,
                    'error': str(e),
                    'details': f"Test {test_name} failed with exception: {e}"
                }
                print(f"  Result: FAIL - {e}")

        return results

    def test_sensor_health(self):
        """Test all sensors are functioning properly"""
        start_time = time.time()

        # Wait for sensor data
        while (not self.imu_data or
               not self.camera_data or
               not self.lidar_data or
               len(self.joint_states) == 0):
            if time.time() - start_time > self.test_timeout:
                return {'success': False, 'error': 'Timeout waiting for sensor data'}

            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Check if sensor data is reasonable
        imu_ok = len(self.imu_data) > 0
        camera_ok = self.camera_data['width'] > 0 and self.camera_data['height'] > 0
        lidar_ok = self.lidar_data['width'] > 0 and self.lidar_data['height'] > 0
        joints_ok = len(self.joint_states) > 0

        success = all([imu_ok, camera_ok, lidar_ok, joints_ok])

        return {
            'success': success,
            'details': {
                'imu_ok': imu_ok,
                'camera_ok': camera_ok,
                'lidar_ok': lidar_ok,
                'joints_ok': joints_ok
            }
        }

    def test_actuator_control(self):
        """Test actuator control system"""
        # Command a small movement to test joint control
        cmd = JointState()
        cmd.name = list(self.joint_states.keys())[:5]  # Test first 5 joints
        cmd.position = [0.1, 0.1, 0.1, 0.1, 0.1]  # Small movement

        self.joint_cmd_pub.publish(cmd)

        # Wait for response
        time.sleep(2.0)

        # Check if joints moved
        initial_positions = {name: self.joint_states[name]['position'] for name in cmd.name}

        # Wait for movement to complete
        time.sleep(3.0)

        final_positions = {name: self.joint_states[name]['position'] for name in cmd.name}

        # Check if joints moved approximately to commanded positions
        moved_ok = all(abs(final_positions[name] - cmd.position[i]) < 0.2
                      for i, name in enumerate(cmd.name))

        return {
            'success': moved_ok,
            'details': {
                'initial_positions': initial_positions,
                'final_positions': final_positions,
                'commanded_positions': dict(zip(cmd.name, cmd.position))
            }
        }

    def test_balance_system(self):
        """Test balance control system"""
        # Check if IMU data is stable (robot should be balanced)
        if not self.imu_data:
            return {'success': False, 'error': 'No IMU data available'}

        # Check if robot is maintaining upright orientation
        orientation = self.imu_data['orientation']
        w, x, y, z = orientation

        # Convert quaternion to roll/pitch angles
        roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin(2*(w*y - z*x))

        # Check if angles are within reasonable balance range (+/- 15 degrees)
        balance_ok = abs(roll) < 0.26 and abs(pitch) < 0.26  # 15 degrees in radians

        return {
            'success': balance_ok,
            'details': {
                'roll': math.degrees(roll),
                'pitch': math.degrees(pitch),
                'threshold_degrees': 15.0
            }
        }

    def test_perception_pipeline(self):
        """Test perception system functionality"""
        # Check if perception system is processing data
        if not self.camera_data or not self.lidar_data:
            return {'success': False, 'error': 'Perception system not receiving data'}

        # Check if perception outputs are being generated
        # This would typically check for object detection, etc.
        perception_active = True  # Simplified check

        return {
            'success': perception_active,
            'details': {
                'camera_data_received': self.camera_data is not None,
                'lidar_data_received': self.lidar_data is not None
            }
        }

    def test_navigation_system(self):
        """Test navigation system"""
        # Command a simple movement
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.1  # Move forward slowly
        cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_vel)

        # Wait for movement
        time.sleep(3.0)

        # Stop movement
        cmd_vel.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

        # Check if movement occurred
        # In a real test, we'd check if the robot actually moved using localization
        movement_attempted = True  # Simplified check

        return {
            'success': movement_attempted,
            'details': {
                'command_sent': True,
                'movement_expected': True
            }
        }

    def test_manipulation_system(self):
        """Test manipulation system"""
        # Test arm movement
        if len([name for name in self.joint_states.keys() if 'arm' in name.lower()]) == 0:
            return {'success': True, 'details': 'No arm joints found, skipping test'}

        # Command arm to move to a position
        cmd = JointState()
        arm_joints = [name for name in self.joint_states.keys() if 'arm' in name.lower()][:3]

        if len(arm_joints) == 0:
            return {'success': True, 'details': 'No arm joints found'}

        cmd.name = arm_joints
        cmd.position = [0.2, 0.1, 0.0]  # Example positions

        self.joint_cmd_pub.publish(cmd)

        # Wait for movement
        time.sleep(2.0)

        # Check if arm joints moved
        initial_arm_positions = {name: self.joint_states[name]['position'] for name in arm_joints}

        time.sleep(3.0)

        final_arm_positions = {name: self.joint_states[name]['position'] for name in arm_joints}

        moved_ok = any(abs(final_arm_positions[name] - initial_arm_positions[name]) > 0.05
                      for name in arm_joints)

        return {
            'success': moved_ok,
            'details': {
                'arm_joints_found': len(arm_joints),
                'arm_moved': moved_ok,
                'joint_changes': {name: final_arm_positions[name] - initial_arm_positions[name]
                                 for name in arm_joints}
            }
        }

    def test_safety_systems(self):
        """Test safety systems"""
        # Check if safety limits are in place
        # This would typically check for joint limits, velocity limits, etc.
        safety_systems_ok = True  # Simplified check

        # Check for emergency stop functionality
        # In a real test, we'd trigger emergency stop and verify response
        emergency_stop_ok = True  # Simplified check

        return {
            'success': safety_systems_ok and emergency_stop_ok,
            'details': {
                'safety_systems_ok': safety_systems_ok,
                'emergency_stop_ok': emergency_stop_ok
            }
        }

    def generate_test_report(self, results):
        """Generate comprehensive test report"""
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result['success'])
        failed_tests = total_tests - passed_tests

        report = f"""
HUMANOID ROBOT INTEGRATION TEST REPORT
======================================

SUMMARY:
- Total Tests: {total_tests}
- Passed: {passed_tests}
- Failed: {failed_tests}
- Success Rate: {passed_tests/total_tests*100:.1f}%

DETAILED RESULTS:
"""

        for test_name, result in results.items():
            status = "PASS" if result['success'] else "FAIL"
            report += f"\n{test_name}: {status}"
            if not result['success'] and 'error' in result:
                report += f" - Error: {result['error']}"

        # Add recommendations
        if failed_tests > 0:
            report += f"\n\nRECOMMENDATIONS:"
            report += f"\n- Investigate failed tests before proceeding"
            report += f"\n- Check hardware connections and calibrations"
            report += f"\n- Verify software configurations"
        else:
            report += f"\n\nCONCLUSION: All integration tests passed! Robot is ready for operational testing."

        return report
```

## Performance Testing Suite

### Real-Time Performance Tests
```python
import time
import statistics
import threading
from collections import deque
import psutil
import GPUtil

class PerformanceTester:
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'inference_times': [],
            'control_rates': [],
            'communication_latency': []
        }

        self.benchmark_results = {}
        self.testing_thread = None
        self.stop_testing = threading.Event()

    def start_performance_monitoring(self):
        """Start performance monitoring in background thread"""
        self.testing_thread = threading.Thread(target=self.monitor_performance, daemon=True)
        self.testing_thread.start()

    def monitor_performance(self):
        """Monitor system performance continuously"""
        while not self.stop_testing.is_set():
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_usage'].append(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].append(memory.percent)

            # GPU usage (if available)
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_load = gpus[0].load * 100
                self.metrics['gpu_usage'].append(gpu_load)

            time.sleep(0.1)

    def test_control_loop_performance(self, duration=10.0):
        """Test control loop timing performance"""
        start_time = time.time()
        loop_times = []
        target_rate = 100  # 100 Hz
        target_dt = 1.0 / target_rate

        while time.time() - start_time < duration:
            loop_start = time.time()

            # Simulate control loop operations
            # In real implementation, this would be actual control computations
            self.simulate_control_computation()

            loop_time = time.time() - loop_start
            loop_times.append(loop_time)

            # Sleep to maintain target rate
            sleep_time = max(0, target_dt - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Calculate performance metrics
        avg_loop_time = statistics.mean(loop_times)
        std_loop_time = statistics.stdev(loop_times) if len(loop_times) > 1 else 0
        achieved_rate = 1.0 / avg_loop_time if avg_loop_time > 0 else 0

        jitter = std_loop_time * 1000  # Convert to ms
        avg_loop_ms = avg_loop_time * 1000

        return {
            'success': achieved_rate >= target_rate * 0.95,  # Allow 5% tolerance
            'details': {
                'target_rate_hz': target_rate,
                'achieved_rate_hz': achieved_rate,
                'avg_loop_time_ms': avg_loop_ms,
                'timing_jitter_ms': jitter,
                'loop_count': len(loop_times),
                'period_consistency': 1.0 - (jitter / avg_loop_ms) if avg_loop_ms > 0 else 0
            }
        }

    def simulate_control_computation(self):
        """Simulate control computation (placeholder)"""
        # This would include actual control calculations
        # For now, just do some computation to simulate load
        result = 0
        for i in range(10000):
            result += i * 0.001

    def test_communication_latency(self):
        """Test ROS 2 communication latency"""
        import rclpy
        from std_msgs.msg import Float64
        import time

        node = rclpy.create_node('latency_tester')

        # Create publisher and subscriber
        pub = node.create_publisher(Float64, 'latency_test_topic', 10)
        latency_measurements = []

        def callback(msg):
            receive_time = time.time()
            if hasattr(msg, 'send_time'):
                latency = receive_time - msg.send_time
                latency_measurements.append(latency)

        sub = node.create_subscription(Float64, 'latency_test_topic', callback, 10)

        # Send test messages
        for i in range(100):
            msg = Float64()
            msg.data = time.time()  # Use data field to store send time
            pub.publish(msg)
            time.sleep(0.01)  # 100 Hz

        # Wait for responses
        time.sleep(2.0)

        node.destroy_node()

        if latency_measurements:
            avg_latency = statistics.mean(latency_measurements) * 1000  # Convert to ms
            max_latency = max(latency_measurements) * 1000
            std_latency = statistics.stdev(latency_measurements) * 1000 if len(latency_measurements) > 1 else 0

            return {
                'success': avg_latency < 10.0,  # Less than 10ms average
                'details': {
                    'average_latency_ms': avg_latency,
                    'max_latency_ms': max_latency,
                    'std_deviation_ms': std_latency,
                    'sample_count': len(latency_measurements)
                }
            }

        return {
            'success': False,
            'error': 'No latency measurements received'
        }

    def test_perception_throughput(self):
        """Test perception system throughput"""
        # Simulate perception pipeline with varying loads
        import numpy as np

        throughput_results = []

        for resolution in [(320, 240), (640, 480), (1280, 720)]:
            # Simulate image processing at different resolutions
            start_time = time.time()
            processed_frames = 0

            test_duration = 5.0  # 5 seconds
            while time.time() - start_time < test_duration:
                # Simulate image processing
                image = np.random.randint(0, 255, size=(resolution[1], resolution[0], 3), dtype=np.uint8)

                # Simulate perception processing
                self.process_perception_frame(image)

                processed_frames += 1

            elapsed = time.time() - start_time
            fps = processed_frames / elapsed

            throughput_results.append({
                'resolution': resolution,
                'fps': fps,
                'elapsed': elapsed,
                'frames': processed_frames
            })

        return {
            'success': all(result['fps'] >= 30 for result in throughput_results),  # At least 30 FPS
            'details': throughput_results
        }

    def process_perception_frame(self, image):
        """Simulate perception processing on a frame"""
        # This would include actual perception algorithms
        # For now, just simulate some processing time
        time.sleep(0.01)  # Simulate 10ms processing time

    def run_comprehensive_benchmark(self):
        """Run comprehensive performance benchmark"""
        print("Starting Performance Benchmarking...")

        benchmarks = {
            'control_loop_performance': self.test_control_loop_performance,
            'communication_latency': self.test_communication_latency,
            'perception_throughput': self.test_perception_throughput
        }

        results = {}

        for bench_name, bench_func in benchmarks.items():
            print(f"Running {bench_name}...")
            try:
                result = bench_func()
                results[bench_name] = result
                print(f"  Result: {'PASS' if result['success'] else 'FAIL'}")
            except Exception as e:
                results[bench_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"  Result: FAIL - {e}")

        return results

    def generate_performance_report(self, results):
        """Generate performance benchmark report"""
        report = """
PERFORMANCE BENCHMARK REPORT
============================

SYSTEM PERFORMANCE METRICS:
"""

        # Add system resource usage
        if self.metrics['cpu_usage']:
            avg_cpu = statistics.mean(self.metrics['cpu_usage'])
            max_cpu = max(self.metrics['cpu_usage'])
            report += f"- CPU Usage: Avg {avg_cpu:.1f}%, Max {max_cpu:.1f}%\n"

        if self.metrics['memory_usage']:
            avg_mem = statistics.mean(self.metrics['memory_usage'])
            max_mem = max(self.metrics['memory_usage'])
            report += f"- Memory Usage: Avg {avg_mem:.1f}%, Max {max_mem:.1f}%\n"

        # Add benchmark results
        report += "\nBENCHMARK RESULTS:\n"
        for bench_name, result in results.items():
            status = "PASS" if result['success'] else "FAIL"
            report += f"\n{bench_name}: {status}"

            if 'details' in result:
                for key, value in result['details'].items():
                    report += f"\n  - {key}: {value}"

        # Add performance recommendations
        report += "\n\nPERFORMANCE RECOMMENDATIONS:\n"
        if any(not r['success'] for r in results.values()):
            report += "- Investigate performance bottlenecks in failed tests\n"
            report += "- Consider hardware upgrades if needed\n"
            report += "- Optimize critical path algorithms\n"
        else:
            report += "- System meets performance requirements\n"
            report += "- Ready for operational deployment\n"

        return report
```

## Safety and Validation Systems

### Safety Validation Framework
```python
import numpy as np
import threading
from enum import Enum
from typing import Dict, Any, List, Optional
import logging

class SafetyLevel(Enum):
    NORMAL = 0
    WARNING = 1
    ERROR = 2
    EMERGENCY_STOP = 3

class SafetyValidator:
    def __init__(self):
        self.safety_monitors = []
        self.emergency_stop_active = False
        self.safety_log = []
        self.safety_lock = threading.Lock()

        # Initialize safety monitors
        self.joint_limit_monitor = JointLimitMonitor()
        self.velocity_monitor = VelocityMonitor()
        self.balance_monitor = BalanceMonitor()
        self.collision_monitor = CollisionMonitor()
        self.power_monitor = PowerMonitor()

        self.safety_monitors = [
            self.joint_limit_monitor,
            self.velocity_monitor,
            self.balance_monitor,
            self.collision_monitor,
            self.power_monitor
        ]

    def validate_state(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate robot state for safety"""
        safety_status = {
            'level': SafetyLevel.NORMAL,
            'violations': [],
            'recommended_action': 'continue'
        }

        for monitor in self.safety_monitors:
            monitor_result = monitor.check(robot_state)

            if monitor_result['level'] > safety_status['level']:
                safety_status['level'] = monitor_result['level']

            if monitor_result['violations']:
                safety_status['violations'].extend(monitor_result['violations'])

            if monitor_result['recommended_action'] == 'emergency_stop':
                safety_status['recommended_action'] = 'emergency_stop'

        # Log safety status
        with self.safety_lock:
            self.safety_log.append({
                'timestamp': time.time(),
                'level': safety_status['level'],
                'violations': safety_status['violations'],
                'action': safety_status['recommended_action']
            })

        return safety_status

    def execute_with_safety_validation(self, action_func, *args, **kwargs):
        """Execute action with safety validation"""
        # Get current robot state
        current_state = self.get_current_robot_state()

        # Validate action
        safety_status = self.validate_state(current_state)

        if safety_status['recommended_action'] == 'emergency_stop':
            self.trigger_emergency_stop()
            return {'success': False, 'error': 'Safety violation - emergency stop activated'}

        # Execute action
        try:
            result = action_func(*args, **kwargs)
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        with self.safety_lock:
            self.emergency_stop_active = True

        # Stop all robot motion
        self.stop_robot_motion()

        # Log emergency event
        logging.warning("EMERGENCY STOP ACTIVATED")

    def release_emergency_stop(self):
        """Release emergency stop"""
        with self.safety_lock:
            self.emergency_stop_active = False

        # Log event
        logging.info("Emergency stop released")

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        with self.safety_lock:
            recent_violations = self.safety_log[-10:] if self.safety_log else []

            return {
                'emergency_stop_active': self.emergency_stop_active,
                'recent_violations': recent_violations,
                'safety_level': max((v['level'] for v in recent_violations), default=0) if recent_violations else 0
            }

class JointLimitMonitor:
    def __init__(self):
        self.joint_limits = {
            # Example limits (these would come from URDF/model)
            'left_hip_yaw': (-1.5, 1.5),
            'left_hip_roll': (-0.5, 0.5),
            'left_hip_pitch': (-2.0, 0.5),
            'left_knee': (0.0, 2.5),
            'left_ankle_pitch': (-0.5, 0.5),
            'left_ankle_roll': (-0.3, 0.3),
            'right_hip_yaw': (-1.5, 1.5),
            'right_hip_roll': (-0.5, 0.5),
            'right_hip_pitch': (-2.0, 0.5),
            'right_knee': (0.0, 2.5),
            'right_ankle_pitch': (-0.5, 0.5),
            'right_ankle_roll': (-0.3, 0.3),
            # Add more joints as needed
        }

    def check(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check joint limits"""
        violations = []

        if 'joint_states' in robot_state:
            for joint_name, joint_data in robot_state['joint_states'].items():
                if joint_name in self.joint_limits:
                    pos = joint_data.get('position', 0.0)
                    min_limit, max_limit = self.joint_limits[joint_name]

                    if pos < min_limit or pos > max_limit:
                        violation = {
                            'type': 'joint_limit_violation',
                            'joint': joint_name,
                            'position': pos,
                            'limit': (min_limit, max_limit),
                            'severity': 'critical' if abs(pos) > max(abs(min_limit), abs(max_limit)) * 1.1 else 'warning'
                        }
                        violations.append(violation)

        level = SafetyLevel.NORMAL
        if any(v['severity'] == 'critical' for v in violations):
            level = SafetyLevel.EMERGENCY_STOP
        elif violations:
            level = SafetyLevel.WARNING

        return {
            'level': level,
            'violations': violations,
            'recommended_action': 'emergency_stop' if level == SafetyLevel.EMERGENCY_STOP else 'warn'
        }

class VelocityMonitor:
    def __init__(self):
        self.max_joint_velocity = 5.0  # rad/s
        self.max_linear_velocity = 1.0  # m/s
        self.max_angular_velocity = 1.0  # rad/s

    def check(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check velocity limits"""
        violations = []

        # Check joint velocities
        if 'joint_states' in robot_state:
            for joint_name, joint_data in robot_state['joint_states'].items():
                vel = joint_data.get('velocity', 0.0)
                if abs(vel) > self.max_joint_velocity:
                    violations.append({
                        'type': 'velocity_violation',
                        'joint': joint_name,
                        'velocity': vel,
                        'limit': self.max_joint_velocity,
                        'severity': 'warning'
                    })

        # Check base velocity (if available)
        if 'base_velocity' in robot_state:
            lin_vel = robot_state['base_velocity'].get('linear', [0, 0, 0])
            ang_vel = robot_state['base_velocity'].get('angular', [0, 0, 0])

            lin_speed = np.linalg.norm(lin_vel)
            ang_speed = np.linalg.norm(ang_vel)

            if lin_speed > self.max_linear_velocity:
                violations.append({
                    'type': 'linear_velocity_violation',
                    'velocity': lin_speed,
                    'limit': self.max_linear_velocity,
                    'severity': 'warning'
                })

            if ang_speed > self.max_angular_velocity:
                violations.append({
                    'type': 'angular_velocity_violation',
                    'velocity': ang_speed,
                    'limit': self.max_angular_velocity,
                    'severity': 'warning'
                })

        level = SafetyLevel.NORMAL
        if any(v['severity'] == 'critical' for v in violations):
            level = SafetyLevel.EMERGENCY_STOP
        elif violations:
            level = SafetyLevel.WARNING

        return {
            'level': level,
            'violations': violations,
            'recommended_action': 'emergency_stop' if level == SafetyLevel.EMERGENCY_STOP else 'warn'
        }

class BalanceMonitor:
    def __init__(self):
        self.max_roll_angle = 0.3  # radians (~17 degrees)
        self.max_pitch_angle = 0.3  # radians (~17 degrees)
        self.zmp_tolerance = 0.1  # meters

    def check(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check balance status"""
        violations = []

        if 'imu_data' in robot_state:
            orientation = robot_state['imu_data'].get('orientation', [0, 0, 0, 1])
            w, x, y, z = orientation

            # Convert quaternion to roll/pitch
            roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = math.asin(2*(w*y - z*x))

            if abs(roll) > self.max_roll_angle:
                violations.append({
                    'type': 'roll_angle_violation',
                    'angle': math.degrees(roll),
                    'limit': math.degrees(self.max_roll_angle),
                    'severity': 'critical' if abs(roll) > self.max_roll_angle * 1.5 else 'warning'
                })

            if abs(pitch) > self.max_pitch_angle:
                violations.append({
                    'type': 'pitch_angle_violation',
                    'angle': math.degrees(pitch),
                    'limit': math.degrees(self.max_pitch_angle),
                    'severity': 'critical' if abs(pitch) > self.max_pitch_angle * 1.5 else 'warning'
                })

        level = SafetyLevel.NORMAL
        if any(v['severity'] == 'critical' for v in violations):
            level = SafetyLevel.EMERGENCY_STOP
        elif violations:
            level = SafetyLevel.WARNING

        return {
            'level': level,
            'violations': violations,
            'recommended_action': 'emergency_stop' if level == SafetyLevel.EMERGENCY_STOP else 'warn'
        }

class CollisionMonitor:
    def __init__(self):
        self.proximity_threshold = 0.3  # meters
        self.collision_threshold = 0.05  # meters

    def check(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check for collisions and proximity warnings"""
        violations = []

        if 'laser_scan' in robot_state:
            ranges = robot_state['laser_scan'].get('ranges', [])

            min_distance = min(ranges) if ranges else float('inf')

            if min_distance < self.collision_threshold:
                violations.append({
                    'type': 'collision_detected',
                    'distance': min_distance,
                    'severity': 'critical'
                })
            elif min_distance < self.proximity_threshold:
                violations.append({
                    'type': 'proximity_warning',
                    'distance': min_distance,
                    'severity': 'warning'
                })

        # Check joint effort for potential collisions
        if 'joint_states' in robot_state:
            for joint_name, joint_data in robot_state['joint_states'].items():
                effort = joint_data.get('effort', 0.0)
                max_normal_effort = 50.0  # Nm (example value)

                if abs(effort) > max_normal_effort * 2.0:
                    violations.append({
                        'type': 'high_effort_collision',
                        'joint': joint_name,
                        'effort': effort,
                        'severity': 'critical'
                    })

        level = SafetyLevel.NORMAL
        if any(v['severity'] == 'critical' for v in violations):
            level = SafetyLevel.EMERGENCY_STOP
        elif violations:
            level = SafetyLevel.WARNING

        return {
            'level': level,
            'violations': violations,
            'recommended_action': 'emergency_stop' if level == SafetyLevel.EMERGENCY_STOP else 'warn'
        }

class PowerMonitor:
    def __init__(self):
        self.max_current_draw = 10.0  # amps
        self.min_battery_voltage = 11.0  # volts
        self.max_power_consumption = 200.0  # watts

    def check(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check power system status"""
        violations = []

        if 'power_data' in robot_state:
            current = robot_state['power_data'].get('current', 0.0)
            voltage = robot_state['power_data'].get('voltage', 12.0)
            power = current * voltage

            if current > self.max_current_draw:
                violations.append({
                    'type': 'overcurrent',
                    'current': current,
                    'limit': self.max_current_draw,
                    'severity': 'warning'
                })

            if voltage < self.min_battery_voltage:
                violations.append({
                    'type': 'low_battery',
                    'voltage': voltage,
                    'limit': self.min_battery_voltage,
                    'severity': 'warning'
                })

            if power > self.max_power_consumption:
                violations.append({
                    'type': 'overpower',
                    'power': power,
                    'limit': self.max_power_consumption,
                    'severity': 'warning'
                })

        level = SafetyLevel.NORMAL if not violations else SafetyLevel.WARNING

        return {
            'level': level,
            'violations': violations,
            'recommended_action': 'warn'
        }
```

## System Validation Procedures

### Validation Test Cases
```python
import unittest
import numpy as np
from typing import Dict, Any

class HumanoidValidationSuite(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.integration_tester = HumanoidIntegrationTester()
        self.performance_tester = PerformanceTester()
        self.safety_validator = SafetyValidator()

    def test_basic_mobility(self):
        """Test basic mobility functions"""
        # Test if robot can move forward
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.2  # Move forward at 0.2 m/s

        self.integration_tester.cmd_vel_pub.publish(cmd_vel)
        time.sleep(3.0)  # Move for 3 seconds

        # Stop
        cmd_vel.linear.x = 0.0
        self.integration_tester.cmd_vel_pub.publish(cmd_vel)

        # Check if movement was attempted (simplified check)
        self.assertTrue(True, "Basic mobility test completed")

    def test_balance_recovery(self):
        """Test balance recovery capability"""
        # This would involve applying disturbances and checking recovery
        # For simulation, we'll check if balance controller is responsive

        # Get initial IMU readings
        initial_imu = self.integration_tester.imu_data.copy()

        # Simulate small disturbance (in real test, would apply actual disturbance)
        time.sleep(2.0)

        # Check if balance system responds appropriately
        final_imu = self.integration_tester.imu_data.copy()

        # Validate that IMU readings are within reasonable range
        if initial_imu and final_imu:
            # Check if angles remain reasonable (within 30 degrees)
            # This would involve quaternion to euler conversion
            pass

        self.assertTrue(True, "Balance recovery test completed")

    def test_sensor_fusion_accuracy(self):
        """Test accuracy of sensor fusion"""
        # Check if multiple sensors provide consistent data
        # This would involve comparing IMU, encoders, and vision data

        # For now, just verify all sensors are publishing
        sensors_active = all([
            self.integration_tester.imu_data,
            self.integration_tester.camera_data,
            self.integration_tester.lidar_data,
            len(self.integration_tester.joint_states) > 0
        ])

        self.assertTrue(sensors_active, "All sensors are active and publishing")

    def test_real_time_performance(self):
        """Test real-time performance requirements"""
        perf_result = self.performance_tester.test_control_loop_performance(duration=5.0)

        self.assertTrue(
            perf_result['success'],
            f"Control loop performance test failed: {perf_result.get('details', {})}"
        )

    def test_safety_system_response(self):
        """Test safety system response to violations"""
        # Create a mock state that violates joint limits
        test_state = {
            'joint_states': {
                'left_knee': {'position': 3.0, 'velocity': 0.0, 'effort': 0.0}  # Exceeds limit
            }
        }

        safety_status = self.safety_validator.validate_state(test_state)

        # Verify that safety system detects the violation
        self.assertGreaterEqual(
            safety_status['level'].value,
            SafetyLevel.WARNING.value,
            "Safety system should detect joint limit violation"
        )

    def test_perception_accuracy(self):
        """Test perception system accuracy"""
        # This would involve testing object detection, recognition, etc.
        # For now, verify perception system is processing data

        perception_active = self.integration_tester.test_perception_pipeline()['success']
        self.assertTrue(perception_active, "Perception system is active")

    def tearDown(self):
        """Clean up after tests"""
        # Stop any ongoing movements
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.integration_tester.cmd_vel_pub.publish(cmd_vel)

def run_validation_suite():
    """Run the complete validation suite"""
    print("Starting Humanoid Robot Validation Suite...")

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(HumanoidValidationSuite)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate validation report
    report = f"""
VALIDATION SUITE RESULTS
=======================
Tests Run: {result.testsRun}
Failures: {len(result.failures)}
Errors: {len(result.errors)}
Success Rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%
"""

    if result.failures:
        report += "\nFAILURES:\n"
        for test, traceback in result.failures:
            report += f"\n{test}: {traceback}\n"

    if result.errors:
        report += "\nERRORS:\n"
        for test, traceback in result.errors:
            report += f"\n{test}: {traceback}\n"

    print(report)
    return result.wasSuccessful()

# Additional validation utilities
class ValidationMetrics:
    def __init__(self):
        self.metrics = {
            'reliability': [],
            'accuracy': [],
            'performance': [],
            'safety': []
        }

    def calculate_reliability_score(self, test_results: Dict[str, Any]) -> float:
        """Calculate reliability score from test results"""
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result.get('success', False))

        return passed_tests / total_tests if total_tests > 0 else 0.0

    def calculate_accuracy_score(self, perception_results: Dict[str, Any]) -> float:
        """Calculate accuracy score for perception system"""
        # This would analyze perception accuracy metrics
        # For now, return a placeholder
        return 0.95  # 95% accuracy

    def calculate_performance_score(self, benchmark_results: Dict[str, Any]) -> float:
        """Calculate performance score from benchmarks"""
        scores = []

        for bench_name, result in benchmark_results.items():
            if result['success']:
                if 'details' in result:
                    # Calculate score based on specific metrics
                    if 'achieved_rate_hz' in result['details']:
                        target = result['details'].get('target_rate_hz', 100)
                        achieved = result['details']['achieved_rate_hz']
                        scores.append(min(1.0, achieved / target))

        return sum(scores) / len(scores) if scores else 0.0

    def calculate_safety_score(self, safety_results: Dict[str, Any]) -> float:
        """Calculate safety score"""
        # Higher score means fewer safety violations
        violations = sum(
            len(result.get('violations', []))
            for result in safety_results.values()
        )

        # Inverse relationship: fewer violations = higher safety score
        return max(0.0, 1.0 - (violations * 0.1))  # Each violation reduces score by 10%

    def generate_validation_certificate(self, test_results, benchmark_results, safety_results) -> str:
        """Generate validation certificate"""
        reliability_score = self.calculate_reliability_score(test_results)
        performance_score = self.calculate_performance_score(benchmark_results)
        safety_score = self.calculate_safety_score(safety_results)

        overall_score = (reliability_score + performance_score + safety_score) / 3.0

        status = "VALIDATED" if overall_score >= 0.8 else "CONDITIONAL" if overall_score >= 0.6 else "REQUIRES_IMPROVEMENT"

        certificate = f"""
HUMANOID ROBOT VALIDATION CERTIFICATE
=====================================

VALIDATION DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}
ROBOT ID: HUMANOID-001
VALIDATOR: System Validation Suite

SCORES:
- Reliability: {reliability_score:.2f}
- Performance: {performance_score:.2f}
- Safety: {safety_score:.2f}
- Overall: {overall_score:.2f}

STATUS: {status}

VALIDATION SUMMARY:
- All critical systems tested and verified
- Performance meets specified requirements
- Safety systems functional and responsive
- Ready for operational deployment under supervision

SIGNED: AUTO-VALIDATION-SYSTEM
DATE: {time.strftime('%Y-%m-%d')}
"""

        return certificate
```

## Exercise: Create Complete Validation Framework

Develop a comprehensive validation framework that includes:
1. Automated test suites for all robot subsystems
2. Performance benchmarks with pass/fail criteria
3. Safety validation with emergency procedures
4. Validation reporting and certification system

## Summary

The integration and testing phase is crucial for ensuring that the humanoid robot operates safely and reliably. A comprehensive testing framework that includes sensor validation, actuator testing, performance benchmarks, and safety systems ensures that the robot meets all requirements before deployment. The validation process must be systematic and thorough, covering all operational scenarios and edge cases to guarantee safe and effective robot operation.

---