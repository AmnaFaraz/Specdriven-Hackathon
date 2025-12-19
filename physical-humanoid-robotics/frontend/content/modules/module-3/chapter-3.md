---
id: module-3-chapter-3
title: "AI Planning and Navigation with Isaac"
sidebar_label: "Isaac Planning"
---

# AI Planning and Navigation with Isaac

This chapter explores how NVIDIA Isaac provides AI-powered planning and navigation capabilities for autonomous robots, with a focus on human-safe navigation for humanoid robots.

## Isaac Navigation System

Isaac Navigation includes:

- **Global Path Planning**: Long-term route planning
- **Local Path Planning**: Obstacle avoidance and dynamic replanning
- **Map Building**: SLAM and semantic mapping
- **Human-Aware Navigation**: Navigation that considers human safety and comfort

## Isaac Navigation Stack

### Global Planner
```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
import numpy as np
import heapq

class IsaacGlobalPlanner(Node):
    def __init__(self):
        super().__init__('isaac_global_planner')

        # Subscriptions
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10
        )

        # Publishers
        self.path_pub = self.create_publisher(Path, '/global_plan', 10)
        self.visualization_pub = self.create_publisher(Marker, '/path_visualization', 10)

        # Navigation data
        self.costmap = None
        self.start_pose = None
        self.goal_pose = None

    def map_callback(self, msg):
        """Process occupancy grid map"""
        self.costmap = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)

    def goal_callback(self, msg):
        """Process navigation goal and plan path"""
        self.goal_pose = msg.pose
        if self.costmap is not None:
            path = self.plan_path(self.start_pose, self.goal_pose)
            self.publish_path(path)

    def plan_path(self, start, goal):
        """Plan path using A* algorithm with Isaac optimizations"""
        # Convert world coordinates to map coordinates
        start_map = self.world_to_map(start.position.x, start.position.y)
        goal_map = self.world_to_map(goal.position.x, goal.position.y)

        # Implement A* path planning algorithm
        open_set = [(0, start_map)]
        came_from = {}
        g_score = {start_map: 0}
        f_score = {start_map: self.heuristic(start_map, goal_map)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_map:
                # Reconstruct path
                path = self.reconstruct_path(came_from, current)
                return self.convert_path_to_ros(path)

            for neighbor in self.get_neighbors(current):
                if self.is_valid_cell(neighbor):
                    tentative_g_score = g_score[current] + self.distance(current, neighbor)

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_map)

                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, a, b):
        """Heuristic function for A* (Euclidean distance)"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, pos):
        """Get valid neighbors for path planning"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((pos[0] + dx, pos[1] + dy))
        return neighbors

    def is_valid_cell(self, pos):
        """Check if cell is valid for navigation"""
        x, y = pos
        if x < 0 or x >= self.costmap.shape[1] or y < 0 or y >= self.costmap.shape[0]:
            return False
        # Check if cell is occupied (value > 50 on 0-100 scale)
        return self.costmap[y, x] < 50

    def world_to_map(self, x_world, y_world):
        """Convert world coordinates to map coordinates"""
        x_map = int((x_world - self.map_origin[0]) / self.map_resolution)
        y_map = int((y_world - self.map_origin[1]) / self.map_resolution)
        return (x_map, y_map)

    def convert_path_to_ros(self, path):
        """Convert path to ROS Path message"""
        ros_path = Path()
        ros_path.header.frame_id = "map"
        ros_path.header.stamp = self.get_clock().now().to_msg()

        for point in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = point[0] * self.map_resolution + self.map_origin[0]
            pose.pose.position.y = point[1] * self.map_resolution + self.map_origin[1]
            ros_path.poses.append(pose)

        return ros_path
```

## Isaac Local Planner and Obstacle Avoidance

### DWA Local Planner
```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformListener, Buffer
import numpy as np

class IsaacLocalPlanner(Node):
    def __init__(self):
        super().__init__('isaac_local_planner')

        # Subscriptions
        self.global_path_sub = self.create_subscription(
            Path, '/global_plan', self.global_path_callback, 10
        )
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)

        # Robot parameters
        self.robot_radius = 0.3  # meters
        self.max_vel_x = 0.5
        self.max_vel_theta = 1.0
        self.min_vel_x = 0.1
        self.min_vel_theta = 0.1

        # Trajectory scoring weights
        self.goal_cost_weight = 1.0
        self.obstacle_cost_weight = 1.0
        self.velocity_cost_weight = 0.1

    def laser_callback(self, msg):
        """Process laser scan for obstacle detection"""
        self.laser_ranges = np.array(msg.ranges)
        self.laser_angle_min = msg.angle_min
        self.laser_angle_max = msg.angle_max
        self.laser_angle_increment = msg.angle_increment

    def global_path_callback(self, msg):
        """Process global path and generate local plan"""
        self.global_path = msg.poses

    def odom_callback(self, msg):
        """Process odometry for robot state"""
        self.robot_pose = msg.pose.pose
        self.robot_twist = msg.twist.twist

    def generate_trajectory(self, vel_x, vel_theta, time_horizon=1.0, dt=0.1):
        """Generate trajectory for given velocities"""
        trajectory = []
        pose = self.robot_pose
        time = 0.0

        while time < time_horizon:
            # Simple motion model
            new_x = pose.position.x + vel_x * dt * np.cos(pose.orientation.z)
            new_y = pose.position.y + vel_x * dt * np.sin(pose.orientation.z)
            new_theta = pose.orientation.z + vel_theta * dt

            # Update pose
            new_pose = Pose()
            new_pose.position.x = new_x
            new_pose.position.y = new_y
            new_pose.orientation.z = new_theta

            trajectory.append(new_pose)
            time += dt

        return trajectory

    def score_trajectory(self, trajectory):
        """Score trajectory based on goal distance, obstacles, and velocity"""
        # Goal distance cost
        if len(trajectory) > 0:
            final_pose = trajectory[-1]
            goal_dist = self.distance_to_goal(final_pose)
            goal_cost = goal_dist
        else:
            goal_cost = float('inf')

        # Obstacle cost
        obstacle_cost = 0
        for pose in trajectory:
            if self.is_in_collision(pose):
                obstacle_cost = float('inf')
                break
            # Add cost based on proximity to obstacles
            min_dist = self.min_distance_to_obstacles(pose)
            if min_dist < 0.5:  # 0.5m safety margin
                obstacle_cost += (0.5 - min_dist) * 10

        # Velocity cost (prefer higher velocities)
        velocity_cost = -(self.max_vel_x - self.current_vel_x) * self.velocity_cost_weight

        total_cost = (self.goal_cost_weight * goal_cost +
                     self.obstacle_cost_weight * obstacle_cost +
                     velocity_cost)

        return total_cost

    def is_in_collision(self, pose):
        """Check if pose is in collision with obstacles"""
        # Check laser ranges for collision at this pose
        robot_x = pose.position.x
        robot_y = pose.position.y

        # Transform laser points to map frame and check collision
        for i, range_val in enumerate(self.laser_ranges):
            if not np.isfinite(range_val):
                continue
            angle = self.laser_angle_min + i * self.laser_angle_increment
            point_x = robot_x + range_val * np.cos(angle)
            point_y = robot_y + range_val * np.sin(angle)

            # Check distance to this obstacle point
            dist = np.sqrt((point_x - robot_x)**2 + (point_y - robot_y)**2)
            if dist < self.robot_radius:
                return True

        return False

    def control_loop(self):
        """Main control loop for local planning"""
        # Generate possible trajectories
        trajectories = []
        for vel_x in np.linspace(self.min_vel_x, self.max_vel_x, 5):
            for vel_theta in np.linspace(-self.max_vel_theta, self.max_vel_theta, 7):
                trajectory = self.generate_trajectory(vel_x, vel_theta)
                score = self.score_trajectory(trajectory)
                trajectories.append((trajectory, score, (vel_x, vel_theta)))

        # Select best trajectory
        best_trajectory = min(trajectories, key=lambda x: x[1])
        best_vel_x, best_vel_theta = best_trajectory[2]

        # Publish velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = best_vel_x
        cmd_vel.angular.z = best_vel_theta
        self.cmd_vel_pub.publish(cmd_vel)
```

## Human-Aware Navigation

### Social Navigation with Isaac
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from people_msgs.msg import People, Person
from visualization_msgs.msg import MarkerArray
import numpy as np

class IsaacSocialNavigation(Node):
    def __init__(self):
        super().__init__('isaac_social_navigation')

        # Subscriptions
        self.people_sub = self.create_subscription(
            People, '/people', self.people_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10
        )

        # Publishers
        self.social_cmd_pub = self.create_publisher(Twist, '/social_cmd_vel', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/social_visualization', 10)

        # Social navigation parameters
        self.personal_space_radius = 0.8  # meters
        self.comfort_zone_radius = 1.2   # meters
        self.respectful_distance = 1.5   # meters

    def people_callback(self, msg):
        """Process detected people for social navigation"""
        self.people_list = msg.people
        self.update_social_navigation()

    def update_social_navigation(self):
        """Update navigation based on people detection"""
        if not self.people_list:
            return

        # Calculate social forces
        repulsion_force = np.array([0.0, 0.0])
        attraction_force = np.array([0.0, 0.0])

        for person in self.people_list:
            person_pos = np.array([person.position.x, person.position.y])
            robot_pos = np.array([self.robot_pose.position.x, self.robot_pose.position.y])

            # Calculate distance to person
            distance = np.linalg.norm(person_pos - robot_pos)

            if distance < self.comfort_zone_radius:
                # Apply repulsion force to maintain personal space
                direction = robot_pos - person_pos
                direction = direction / np.linalg.norm(direction)  # normalize
                strength = (self.comfort_zone_radius - distance) / self.comfort_zone_radius
                repulsion_force += direction * strength

        # Combine with goal-seeking behavior
        goal_direction = self.calculate_goal_direction()
        combined_force = 0.7 * goal_direction + 0.3 * repulsion_force

        # Normalize and convert to velocity command
        if np.linalg.norm(combined_force) > 0:
            normalized_force = combined_force / np.linalg.norm(combined_force)
            self.publish_social_command(normalized_force)

    def calculate_goal_direction(self):
        """Calculate direction towards navigation goal"""
        if self.goal_pose:
            goal_vec = np.array([self.goal_pose.position.x - self.robot_pose.position.x,
                               self.goal_pose.position.y - self.robot_pose.position.y])
            if np.linalg.norm(goal_vec) > 0:
                return goal_vec / np.linalg.norm(goal_vec)
        return np.array([0.0, 0.0])

    def publish_social_command(self, direction):
        """Publish social navigation command"""
        cmd_vel = Twist()
        cmd_vel.linear.x = min(0.3, np.linalg.norm(direction) * 0.5)  # Speed limit
        cmd_vel.angular.z = np.arctan2(direction[1], direction[0]) * 0.5  # Turn towards direction
        self.social_cmd_pub.publish(cmd_vel)
```

## Isaac SLAM and Mapping

### Semantic SLAM
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
import numpy as np

class IsaacSemanticSLAM(Node):
    def __init__(self):
        super().__init__('isaac_semantic_slam')

        # Subscriptions
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_rect_color', self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/semantic_map', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/slam_pose', 10)

        # SLAM data
        self.keyframes = []
        self.semantic_annotations = {}
        self.current_pose = None

    def rgb_callback(self, msg):
        """Process RGB image for visual SLAM"""
        # Extract features and match with previous frames
        features = self.extract_features(msg)
        self.process_visual_odometry(features, msg.header.stamp)

    def depth_callback(self, msg):
        """Process depth image for 3D reconstruction"""
        # Combine with RGB for semantic mapping
        if self.current_pose:
            self.update_3d_map(msg, self.current_pose)

    def process_visual_odometry(self, features, timestamp):
        """Process visual odometry for pose estimation"""
        # Match features with previous keyframes
        # Estimate camera motion
        # Update robot pose
        pass

    def update_3d_map(self, depth_msg, pose):
        """Update 3D semantic map with new observations"""
        # Convert depth image to 3D points in robot frame
        # Transform to global map frame using pose
        # Integrate into occupancy grid
        # Add semantic labels from perception system
        pass

    def integrate_semantic_labels(self, segmentation_result, pose):
        """Integrate semantic segmentation results into map"""
        # Project segmentation onto 3D map
        # Update semantic labels for map regions
        # Handle label conflicts and uncertainties
        pass
```

## Exercise: Implement Social Navigation

Create a navigation system that:
1. Detects people in the environment
2. Plans paths that respect personal space
3. Adjusts speed and behavior based on social context
4. Visualizes social navigation decisions

## Summary

Isaac provides comprehensive AI planning and navigation capabilities that enable robots to move safely and efficiently in human environments. The combination of global planning, local obstacle avoidance, and social navigation makes it possible to create robots that can navigate complex environments while respecting human comfort and safety.

---