---
id: module-3-chapter-2
title: "Isaac ROS Perception Pipeline"
sidebar_label: "Isaac Perception"
---

# Isaac ROS Perception Pipeline

This chapter explores the perception capabilities of NVIDIA Isaac ROS, focusing on how AI-powered perception systems enable robots to understand their environment.

## Isaac ROS Perception Stack

The Isaac ROS perception stack includes:

- **Visual Perception**: Object detection, segmentation, tracking
- **Depth Perception**: Stereo vision, depth estimation, 3D reconstruction
- **Sensor Fusion**: Combining multiple sensor modalities
- **AI Inference**: Optimized neural network execution

## Isaac ROS Perception Nodes

### AprilTag Detection
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
import numpy as np

class IsaacAprilTagProcessor(Node):
    def __init__(self):
        super().__init__('isaac_apriltag_processor')

        # Subscribe to camera data
        self.image_sub = self.create_subscription(
            Image, '/camera/image_rect_color', self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
        )

        # Subscribe to AprilTag detections
        self.detection_sub = self.create_subscription(
            AprilTagDetectionArray, '/apriltag_detections', self.detection_callback, 10
        )

        # Publisher for processed poses
        self.pose_pub = self.create_publisher(PoseStamped, '/apriltag_poses', 10)

        self.camera_intrinsics = None

    def camera_info_callback(self, msg):
        """Store camera intrinsics for pose calculation"""
        self.camera_intrinsics = np.array(msg.k).reshape(3, 3)

    def detection_callback(self, msg):
        """Process AprilTag detections and calculate poses"""
        for detection in msg.detections:
            if self.camera_intrinsics is not None:
                # Calculate pose relative to camera
                pose = self.calculate_pose(detection, self.camera_intrinsics)

                # Publish pose
                pose_msg = PoseStamped()
                pose_msg.header = msg.header
                pose_msg.pose = pose
                self.pose_pub.publish(pose_msg)

    def calculate_pose(self, detection, intrinsics):
        """Calculate pose from AprilTag detection"""
        # Implementation would use PnP algorithm with known tag size
        # and camera intrinsics
        pass
```

### Image Segmentation
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from isaac_ros_segmentation_interfaces.msg import SegmentationTensorRT
import numpy as np
import cv2

class IsaacSegmentationProcessor(Node):
    def __init__(self):
        super().__init__('isaac_segmentation_processor')

        self.segmentation_sub = self.create_subscription(
            SegmentationTensorRT,
            '/segmentation/segmentation_map',
            self.segmentation_callback,
            10
        )

        self.visualization_pub = self.create_publisher(
            Image,
            '/segmentation/visualization',
            10
        )

        # Class mapping for visualization
        self.class_colors = {
            0: [0, 0, 0],      # Background
            1: [255, 0, 0],    # Person
            2: [0, 255, 0],    # Obstacle
            3: [0, 0, 255],    # Furniture
            4: [255, 255, 0],  # Door
        }

    def segmentation_callback(self, msg):
        """Process segmentation results and create visualization"""
        # Reshape segmentation data
        seg_map = np.array(msg.tensor.data).reshape((msg.height, msg.width))

        # Create color visualization
        vis_image = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)

        for class_id, color in self.class_colors.items():
            mask = (seg_map == class_id)
            vis_image[mask] = color

        # Convert to ROS image message
        vis_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
        vis_msg.header = msg.header
        self.visualization_pub.publish(vis_msg)

        # Process each class separately
        self.process_persons(seg_map)
        self.process_obstacles(seg_map)

    def process_persons(self, seg_map):
        """Process person detections from segmentation"""
        person_mask = (seg_map == 1)
        if np.any(person_mask):
            # Find contours of person regions
            contours, _ = cv2.findContours(
                person_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small regions
                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    # Process person detection
                    self.handle_person_detection(x, y, w, h)

    def process_obstacles(self, seg_map):
        """Process obstacle detections from segmentation"""
        obstacle_mask = (seg_map == 2)
        if np.any(obstacle_mask):
            # Create occupancy grid or navigation costmap
            self.update_navigation_map(obstacle_mask)
```

## Isaac ROS Depth Perception

### Stereo Disparity Processing
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
import numpy as np

class IsaacDepthProcessor(Node):
    def __init__(self):
        super().__init__('isaac_depth_processor')

        self.disparity_sub = self.create_subscription(
            DisparityImage,
            '/stereo/disparity',
            self.disparity_callback,
            10
        )

        self.depth_pub = self.create_publisher(
            Image,
            '/stereo/depth',
            10
        )

        self.bridge = CvBridge()

    def disparity_callback(self, msg):
        """Convert disparity to depth"""
        # Convert disparity image to numpy array
        disparity_img = self.bridge.imgmsg_to_cv2(msg.image)

        # Calculate depth from disparity
        # depth = (baseline * focal_length) / disparity
        baseline = msg.t.max_disparity  # This is a simplification
        focal_length = msg.f  # From camera calibration

        # Avoid division by zero
        depth_img = np.zeros_like(disparity_img, dtype=np.float32)
        valid_mask = disparity_img > 0
        depth_img[valid_mask] = (baseline * focal_length) / disparity_img[valid_mask]

        # Publish depth image
        depth_msg = self.bridge.cv2_to_imgmsg(depth_img, "32FC1")
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)
```

## Isaac ROS Sensor Fusion

Combining multiple sensor modalities for robust perception:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import TransformListener, Buffer
import numpy as np

class IsaacSensorFusion(Node):
    def __init__(self):
        super().__init__('isaac_sensor_fusion')

        # Initialize TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to multiple sensors
        self.image_sub = self.create_subscription(
            Image, '/camera/image_rect_color', self.image_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/velodyne_points', self.lidar_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # Data storage
        self.latest_image = None
        self.latest_lidar = None
        self.latest_scan = None

        # Fusion timer
        self.fusion_timer = self.create_timer(0.1, self.fusion_callback)

    def image_callback(self, msg):
        """Store latest image"""
        self.latest_image = msg

    def lidar_callback(self, msg):
        """Process LiDAR data"""
        self.latest_lidar = msg
        # Convert PointCloud2 to usable format for fusion

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.latest_scan = msg
        # Process laser scan for immediate obstacle detection

    def fusion_callback(self):
        """Fuse sensor data for comprehensive perception"""
        if self.latest_image and self.latest_lidar:
            # Project image data to 3D using depth
            # Combine with LiDAR data
            # Create fused perception output
            fused_perception = self.fuse_image_lidar(
                self.latest_image,
                self.latest_lidar
            )

            # Publish fused results
            self.publish_fused_data(fused_perception)

    def fuse_image_lidar(self, image_msg, lidar_msg):
        """Fuse image and LiDAR data"""
        # Convert image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # Extract 3D points from LiDAR
        points_3d = self.extract_lidar_points(lidar_msg)

        # Project 3D points to image plane
        projected_points = self.project_3d_to_2d(points_3d, image_msg.header.frame_id)

        # Combine semantic information from image with geometric data from LiDAR
        fused_data = {
            'semantic_map': self.get_semantic_info(cv_image),
            'geometric_map': self.get_geometric_info(points_3d),
            'combined_map': self.combine_maps(cv_image, points_3d)
        }

        return fused_data
```

## Performance Optimization

### TensorRT Optimization
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from isaac_ros_tensor_rt_interfaces.msg import EngineInfo
import tensorrt as trt

class OptimizedInferenceNode(Node):
    def __init__(self):
        super().__init__('optimized_inference')

        # Create TensorRT engine
        self.trt_engine = self.load_trt_engine()

        # Set up optimized pipeline
        self.setup_optimized_pipeline()

    def load_trt_engine(self):
        """Load optimized TensorRT engine"""
        # Load pre-built TensorRT engine
        with open('model.plan', 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)

        return engine

    def setup_optimized_pipeline(self):
        """Setup GPU-optimized processing pipeline"""
        # Configure GPU memory pools
        # Set up CUDA streams for parallel processing
        # Optimize data transfers between CPU and GPU
        pass
```

## Exercise: Build a Multi-Modal Perception System

Create a perception system that:
1. Subscribes to camera, LiDAR, and IMU data
2. Performs object detection using Isaac ROS
3. Fuses sensor data for robust perception
4. Publishes a comprehensive environment model

## Summary

Isaac ROS provides powerful perception capabilities that enable robots to understand their environment through AI-powered processing. The GPU acceleration and optimized neural networks make real-time perception feasible for complex robotic applications. Proper sensor fusion techniques combine multiple modalities for robust and reliable perception.

---