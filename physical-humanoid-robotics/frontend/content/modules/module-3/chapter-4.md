---
id: module-3-chapter-4
title: "Isaac Manipulation and Control"
sidebar_label: "Isaac Manipulation"
---

# Isaac Manipulation and Control

This chapter explores how NVIDIA Isaac enables sophisticated manipulation and control capabilities for robotic systems, particularly relevant for humanoid robots with complex manipulation requirements.

## Isaac Manipulation Framework

Isaac Manipulation includes:

- **Motion Planning**: Trajectory generation for complex manipulation tasks
- **Grasping**: AI-powered grasp planning and execution
- **Force Control**: Precise force and torque control for delicate operations
- **Multi-Arm Coordination**: Control of multiple manipulator arms

## Isaac Motion Planning

### Trajectory Generation
```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped, Point
from moveit_msgs.msg import MoveItErrorCodes
from sensor_msgs.msg import JointState
import numpy as np
from scipy.interpolate import interp1d

class IsaacMotionPlanner(Node):
    def __init__(self):
        super().__init__('isaac_motion_planner')

        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, '/manipulation_goal', self.goal_pose_callback, 10
        )

        # Publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory', 10
        )

        # Robot parameters
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.current_joint_positions = [0.0] * len(self.joint_names)

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_joint_positions[i] = msg.position[idx]

    def plan_trajectory(self, start_pos, goal_pos, duration=5.0):
        """Plan smooth trajectory between start and goal positions"""
        # Use cubic spline interpolation for smooth motion
        t_points = np.array([0, duration/3, 2*duration/3, duration])
        positions = np.array([start_pos,
                             (np.array(start_pos) + np.array(goal_pos))/2,  # Midpoint
                             (np.array(start_pos) + 2*np.array(goal_pos))/3,  # Weighted toward goal
                             goal_pos])

        # Create interpolated functions for each joint
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        # Generate trajectory points
        num_points = int(duration * 50)  # 50Hz trajectory
        time_step = duration / num_points

        for i in range(num_points + 1):
            t = i * time_step

            # Interpolate position for each joint
            point = JointTrajectoryPoint()
            point.positions = []

            for j in range(len(self.joint_names)):
                # Cubic interpolation
                x = t_points
                y = positions[:, j]

                # Fit cubic polynomial
                coeffs = np.polyfit(x, y, 3)
                pos = np.polyval(coeffs, t)

                point.positions.append(pos)

            # Calculate velocities (derivative of position)
            point.velocities = []
            for j in range(len(self.joint_names)):
                coeffs = np.polyfit(t_points, positions[:, j], 3)
                # Derivative: 3*a*t^2 + 2*b*t + c
                vel = 3*coeffs[0]*t**2 + 2*coeffs[1]*t + coeffs[2]
                point.velocities.append(vel)

            # Calculate accelerations (second derivative)
            point.accelerations = []
            for j in range(len(self.joint_names)):
                coeffs = np.polyfit(t_points, positions[:, j], 3)
                # Second derivative: 6*a*t + 2*b
                acc = 6*coeffs[0]*t + 2*coeffs[1]
                point.accelerations.append(acc)

            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)

            trajectory.points.append(point)

        return trajectory

    def execute_trajectory(self, trajectory):
        """Execute the planned trajectory"""
        self.trajectory_pub.publish(trajectory)
```

## Isaac Grasping System

### AI-Powered Grasp Planning
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Bool
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from visualization_msgs.msg import MarkerArray
import numpy as np
import open3d as o3d

class IsaacGraspPlanner(Node):
    def __init__(self):
        super().__init__('isaac_grasp_planner')

        # Subscriptions
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10
        )
        self.detection_sub = self.create_subscription(
            AprilTagDetectionArray, '/apriltag_detections', self.detection_callback, 10
        )

        # Publishers
        self.grasp_poses_pub = self.create_publisher(PoseArray, '/grasp_poses', 10)
        self.grasp_command_pub = self.create_publisher(Bool, '/grasp_command', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/grasp_visualization', 10)

        self.pointcloud_data = None
        self.object_poses = []

    def pointcloud_callback(self, msg):
        """Process point cloud for grasp planning"""
        # Convert PointCloud2 to numpy array
        self.pointcloud_data = self.pointcloud2_to_array(msg)

        # Segment objects from point cloud
        segmented_objects = self.segment_objects(self.pointcloud_data)

        # Plan grasps for each object
        for obj in segmented_objects:
            grasps = self.plan_grasps_for_object(obj)
            self.publish_grasp_candidates(grasps)

    def segment_objects(self, pointcloud):
        """Segment individual objects from point cloud"""
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])

        # Apply statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Plane segmentation to remove ground
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )

        # Remove ground plane points
        object_cloud = pcd.select_by_index(inliers, invert=True)

        # Cluster remaining points to identify objects
        labels = np.array(object_cloud.cluster_dbscan(eps=0.02, min_points=10))

        # Group points by cluster
        objects = []
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
            mask = labels == label
            obj_points = np.asarray(object_cloud.points)[mask]
            objects.append(obj_points)

        return objects

    def plan_grasps_for_object(self, object_points):
        """Plan potential grasps for an object"""
        grasps = []

        # Calculate object properties
        centroid = np.mean(object_points, axis=0)
        size = np.max(object_points, axis=0) - np.min(object_points, axis=0)

        # Generate grasp candidates around the object
        for angle in np.linspace(0, 2*np.pi, 8):
            for height_ratio in [0.3, 0.5, 0.7]:  # Different grasp heights
                grasp_pose = self.calculate_grasp_pose(
                    centroid, size, angle, height_ratio
                )

                # Score the grasp based on geometric properties
                score = self.score_grasp(grasp_pose, object_points)

                if score > 0.5:  # Threshold for valid grasp
                    grasps.append({
                        'pose': grasp_pose,
                        'score': score,
                        'approach_direction': self.calculate_approach_direction(grasp_pose)
                    })

        # Sort grasps by score
        grasps.sort(key=lambda x: x['score'], reverse=True)
        return grasps[:5]  # Return top 5 grasps

    def calculate_grasp_pose(self, centroid, size, angle, height_ratio):
        """Calculate grasp pose based on object properties"""
        pose = Pose()

        # Position grasp at object centroid with height offset
        pose.position.x = centroid[0]
        pose.position.y = centroid[1]
        pose.position.z = centroid[2] + size[2] * height_ratio

        # Orient gripper to approach from the side
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = np.sin(angle / 2)
        pose.orientation.w = np.cos(angle / 2)

        return pose

    def score_grasp(self, pose, object_points):
        """Score grasp based on geometric feasibility"""
        # Check if grasp is collision-free
        collision_score = self.check_grasp_collision(pose, object_points)

        # Check grasp stability
        stability_score = self.evaluate_grasp_stability(pose, object_points)

        # Combine scores
        total_score = 0.6 * stability_score + 0.4 * collision_score

        return total_score

    def execute_grasp(self, grasp_pose):
        """Execute the selected grasp"""
        # Move to pre-grasp position
        pre_grasp = self.calculate_pre_grasp_pose(grasp_pose)
        self.move_to_pose(pre_grasp)

        # Approach the object
        self.move_to_pose(grasp_pose)

        # Close gripper
        self.close_gripper()

        # Lift object
        self.lift_object()

        # Move to destination
        # (Implementation would include collision checking and path planning)
```

## Isaac Force Control

### Impedance Control for Safe Interaction
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class IsaacImpedanceController(Node):
    def __init__(self):
        super().__init__('isaac_impedance_controller')

        # Subscriptions
        self.wrench_sub = self.create_subscription(
            WrenchStamped, '/wrench', self.wrench_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Publishers
        self.impedance_cmd_pub = self.create_publisher(
            Float64MultiArray, '/impedance_command', 10
        )
        self.velocity_cmd_pub = self.create_publisher(Twist, '/velocity_command', 10)

        # Impedance control parameters
        self.stiffness = np.diag([1000, 1000, 1000, 100, 100, 100])  # [x,y,z,R,P,Y]
        self.damping = np.diag([100, 100, 100, 10, 10, 10])
        self.mass = np.diag([10, 10, 10, 1, 1, 1])

        # Force limits for safety
        self.max_force = 50.0  # Newtons
        self.max_torque = 5.0  # Nm

        self.current_wrench = np.zeros(6)  # [fx, fy, fz, mx, my, mz]
        self.current_velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]

    def wrench_callback(self, msg):
        """Update current wrench measurements"""
        self.current_wrench = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])

    def joint_state_callback(self, msg):
        """Update joint state information"""
        # Process joint positions, velocities, efforts
        pass

    def compute_impedance_force(self, desired_pose, current_pose):
        """Compute impedance-based force command"""
        # Calculate pose error
        pose_error = desired_pose - current_pose

        # Apply impedance model: F = K * x + D * v + M * a
        # For simplicity, using only stiffness term
        impedance_force = np.dot(self.stiffness, pose_error[:6])

        # Limit force for safety
        force_magnitude = np.linalg.norm(impedance_force[:3])
        torque_magnitude = np.linalg.norm(impedance_force[3:])

        if force_magnitude > self.max_force:
            impedance_force[:3] = (impedance_force[:3] / force_magnitude) * self.max_force

        if torque_magnitude > self.max_torque:
            impedance_force[3:] = (impedance_force[3:] / torque_magnitude) * self.max_torque

        return impedance_force

    def safety_check(self, force_command):
        """Check if force command is safe to execute"""
        force_mag = np.linalg.norm(force_command[:3])
        torque_mag = np.linalg.norm(force_command[3:])

        if force_mag > self.max_force or torque_mag > self.max_torque:
            self.get_logger().warn(f"Force command exceeds safety limits: F={force_mag}, T={torque_mag}")
            return False

        return True

    def control_loop(self):
        """Main impedance control loop"""
        # Get desired pose from higher-level planner
        desired_pose = self.get_desired_pose()
        current_pose = self.get_current_pose()

        # Compute impedance force
        impedance_force = self.compute_impedance_force(desired_pose, current_pose)

        # Add compliance to external forces
        compliant_force = impedance_force - 0.1 * self.current_wrench  # Damping effect

        # Safety check
        if self.safety_check(compliant_force):
            # Publish impedance command
            cmd_msg = Float64MultiArray()
            cmd_msg.data = compliant_force.tolist()
            self.impedance_cmd_pub.publish(cmd_msg)
```

## Isaac Multi-Arm Coordination

### Coordinated Manipulation
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import numpy as np

class IsaacMultiArmCoordinator(Node):
    def __init__(self):
        super().__init__('isaac_multi_arm_coordinator')

        # Subscriptions for both arms
        self.left_joint_sub = self.create_subscription(
            JointState, '/left_arm/joint_states', self.left_joint_callback, 10
        )
        self.right_joint_sub = self.create_subscription(
            JointState, '/right_arm/joint_states', self.right_joint_callback, 10
        )

        self.left_ee_sub = self.create_subscription(
            Pose, '/left_arm/ee_pose', self.left_ee_callback, 10
        )
        self.right_ee_sub = self.create_subscription(
            Pose, '/right_arm/ee_pose', self.right_ee_callback, 10
        )

        # Publishers
        self.left_cmd_pub = self.create_publisher(JointTrajectory, '/left_arm/command', 10)
        self.right_cmd_pub = self.create_publisher(JointTrajectory, '/right_arm/command', 10)
        self.coordinated_status_pub = self.create_publisher(Bool, '/coordinated_status', 10)

        # Arm states
        self.left_joint_pos = []
        self.right_joint_pos = []
        self.left_ee_pose = None
        self.right_ee_pose = None

    def left_joint_callback(self, msg):
        """Update left arm joint state"""
        self.left_joint_pos = list(msg.position)

    def right_joint_callback(self, msg):
        """Update right arm joint state"""
        self.right_joint_pos = list(msg.position)

    def left_ee_callback(self, msg):
        """Update left arm end-effector pose"""
        self.left_ee_pose = msg

    def right_ee_callback(self, msg):
        """Update right arm end-effector pose"""
        self.right_ee_pose = msg

    def coordinate_arms_for_task(self, task_type):
        """Coordinate both arms for specific tasks"""
        if task_type == "lifting":
            return self.coordinate_for_lifting()
        elif task_type == "assembly":
            return self.coordinate_for_assembly()
        elif task_type == "transport":
            return self.coordinate_for_transport()
        else:
            return self.default_coordination()

    def coordinate_for_lifting(self):
        """Coordinate arms for lifting heavy objects"""
        # Calculate optimal grasp positions
        if self.left_ee_pose and self.right_ee_pose:
            # Calculate center of mass and grasp positions
            center_pos = self.calculate_center_of_mass()

            # Position arms on opposite sides of object
            left_target = self.calculate_grasp_position(center_pos, "left")
            right_target = self.calculate_grasp_position(center_pos, "right")

            # Plan coordinated trajectories
            left_traj = self.plan_coordinated_trajectory("left", left_target)
            right_traj = self.plan_coordinated_trajectory("right", right_target)

            # Ensure synchronized execution
            self.execute_synchronized_trajectories(left_traj, right_traj)

    def calculate_center_of_mass(self):
        """Calculate object center of mass for coordinated lifting"""
        # This would use perception data to estimate object COM
        # For now, return midpoint between end-effectors
        if self.left_ee_pose and self.right_ee_pose:
            center_x = (self.left_ee_pose.position.x + self.right_ee_pose.position.x) / 2
            center_y = (self.left_ee_pose.position.y + self.right_ee_pose.position.y) / 2
            center_z = (self.left_ee_pose.position.z + self.right_ee_pose.position.z) / 2
            return np.array([center_x, center_y, center_z])
        return np.array([0, 0, 0])

    def execute_synchronized_trajectories(self, left_traj, right_traj):
        """Execute trajectories with synchronization"""
        # Ensure both arms start simultaneously
        start_time = self.get_clock().now()

        # Execute trajectories with coordination
        self.left_cmd_pub.publish(left_traj)
        self.right_cmd_pub.publish(right_traj)

        # Monitor for coordination errors
        self.monitor_coordination()

    def monitor_coordination(self):
        """Monitor coordination between arms"""
        if self.left_ee_pose and self.right_ee_pose:
            # Calculate distance between end-effectors
            dist = self.calculate_distance(self.left_ee_pose, self.right_ee_pose)

            # Check if within expected coordination bounds
            if dist > self.max_coordination_distance:
                self.get_logger().warn("Arms coordination error: distance too large")
                self.emergency_stop()
```

## Exercise: Implement Coordinated Grasping

Create a system that:
1. Detects objects in the environment
2. Plans coordinated grasps for dual-arm robot
3. Executes synchronized manipulation
4. Monitors for safety and coordination errors

## Summary

Isaac provides sophisticated manipulation and control capabilities that enable complex robotic tasks. From motion planning to grasp execution, force control, and multi-arm coordination, these capabilities are essential for humanoid robots that need to interact with their environment in meaningful ways.

---