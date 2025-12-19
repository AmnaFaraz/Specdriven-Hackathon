---
id: module-5-chapter-3
title: "AI Perception and Decision Making"
sidebar_label: "AI Perception"
---

# AI Perception and Decision Making

This chapter explores the AI systems that enable the humanoid robot to perceive its environment, understand complex situations, and make intelligent decisions based on multimodal input.

## Perception Architecture

The humanoid robot's perception system processes multiple sensory modalities to build a comprehensive understanding of its environment:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Perception Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│  Sensory Processing Layer                                        │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐ │
│  │   Visual        │   Auditory      │   Tactile & Proprio     │ │
│  │   (Cameras,     │   (Microphones, │   (IMU, Joint Encoders, │ │
│  │   Depth, LIDAR) │   Speakers)     │   Force Sensors)       │ │
│  └─────────────────┴─────────────────┴─────────────────────────┘ │
│                                                                 │
│  Feature Extraction Layer                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Object Detection  │  Semantic Mapping  │  Activity Recog  │ │
│  │  Recognition       │  Scene Understanding│  Human Behavior  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Scene Understanding Layer                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  3D Reconstruction │  Spatial Reasoning  │  Temporal Model  │ │
│  │  SLAM & Mapping   │  Navigation Planning │  Event Tracking  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Decision Making Layer                                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Task Planning      │  Action Selection   │  Behavior       │ │
│  │  Motion Planning    │  Policy Learning    │  Coordination   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Computer Vision for Robotics

### Multi-Modal Vision Processing
```python
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPProcessor
import cv2

class MultiModalVisionProcessor(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        # CLIP for vision-language alignment
        self.clip_vision = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Feature extraction layers
        self.visual_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Object detection head
        self.object_detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 80)  # COCO dataset classes
        )

        # Semantic segmentation head
        self.semantic_segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 50, kernel_size=4, stride=2, padding=1),  # 50 semantic classes
            nn.Sigmoid()
        )

        # Feature projection for fusion
        self.feature_projection = nn.Linear(512, 768)  # Project to CLIP dimension

    def forward(self, pixel_values, texts=None):
        """Process visual input and extract features"""
        # Extract visual features
        vision_features = self.clip_vision(pixel_values=pixel_values)
        vision_features = vision_features.pooler_output
        vision_features = self.feature_projection(vision_features)
        vision_features = nn.functional.normalize(vision_features, dim=-1)

        # Object detection
        object_logits = self.object_detection_head(vision_features)

        # Semantic segmentation
        visual_features_for_seg = self.visual_feature_extractor(pixel_values)
        segmentation_maps = self.semantic_segmentation_head(visual_features_for_seg)

        results = {
            'vision_features': vision_features,
            'object_detection': object_logits,
            'segmentation_maps': segmentation_maps,
            'clip_features': vision_features
        }

        # If texts provided, compute vision-language alignment
        if texts is not None:
            inputs = self.clip_processor(text=texts, images=[None]*len(texts), return_tensors="pt", padding=True)
            text_outputs = self.clip_vision.get_text_features(input_ids=inputs['input_ids'])
            text_features = nn.functional.normalize(text_outputs, dim=-1)

            # Compute similarity
            similarity = torch.matmul(vision_features, text_features.t())
            results['alignment_scores'] = similarity

        return results

    def detect_objects(self, image_tensor):
        """Detect objects in the image"""
        with torch.no_grad():
            results = self.forward(image_tensor)
            object_probs = torch.softmax(results['object_detection'], dim=-1)
            return object_probs

    def segment_scene(self, image_tensor):
        """Perform semantic segmentation of the scene"""
        with torch.no_grad():
            results = self.forward(image_tensor)
            return results['segmentation_maps']

    def align_with_text(self, image_tensor, text_queries):
        """Align visual features with text queries"""
        with torch.no_grad():
            results = self.forward(image_tensor, text_queries)
            return results['alignment_scores']
```

### 3D Scene Understanding
```python
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import distance
import open3d as o3d

class SceneUnderstanding3D(nn.Module):
    def __init__(self):
        super().__init__()

        # 3D feature extraction
        self.pointnet = PointNet3D()
        self.voxel_encoder = VoxelEncoder()
        self.surface_normal_estimator = SurfaceNormalEstimator()

        # Spatial reasoning
        self.spatial_reasoner = SpatialReasoningNetwork()

        # Object relationship modeling
        self.relationship_predictor = RelationshipPredictor()

    def forward(self, point_cloud, camera_intrinsics):
        """Process 3D point cloud for scene understanding"""
        # Extract 3D features
        point_features = self.pointnet(point_cloud)
        voxel_features = self.voxel_encoder(point_cloud)

        # Estimate surface normals
        surface_normals = self.surface_normal_estimator(point_cloud)

        # Extract object proposals
        object_proposals = self.extract_object_proposals(point_cloud, point_features)

        # Compute spatial relationships
        spatial_relations = self.spatial_reasoner.compute_relationships(
            object_proposals, camera_intrinsics
        )

        # Predict object relationships
        relationships = self.relationship_predictor(
            object_proposals, spatial_relations
        )

        return {
            'point_features': point_features,
            'voxel_features': voxel_features,
            'surface_normals': surface_normals,
            'object_proposals': object_proposals,
            'spatial_relations': spatial_relations,
            'relationships': relationships
        }

    def extract_object_proposals(self, point_cloud, features):
        """Extract object proposals from point cloud"""
        # Use clustering algorithms to group points into objects
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())

        # Apply DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10))

        # Group points by cluster
        objects = []
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
            mask = labels == label
            obj_points = np.asarray(pcd.points)[mask]
            obj_feature = features[mask].mean(dim=0)
            obj_center = obj_points.mean(dim=0)

            objects.append({
                'points': obj_points,
                'features': obj_feature,
                'center': obj_center,
                'label': label
            })

        return objects

    def compute_spatial_relations(self, objects, camera_intrinsics):
        """Compute spatial relationships between objects"""
        relations = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    # Compute distance and direction
                    dist = torch.norm(obj1['center'] - obj2['center'])
                    direction = (obj2['center'] - obj1['center']) / dist

                    # Project to camera frame to get spatial relationship
                    rel_type = self.classify_spatial_relation(
                        obj1['center'], obj2['center'], camera_intrinsics
                    )

                    relations.append({
                        'subject': i,
                        'object': j,
                        'distance': dist,
                        'direction': direction,
                        'relation_type': rel_type
                    })

        return relations

    def classify_spatial_relation(self, obj1_center, obj2_center, camera_intrinsics):
        """Classify spatial relationship between objects"""
        # Convert to camera coordinates
        rel_vector = obj2_center - obj1_center

        # Determine spatial relationship based on direction
        if rel_vector[2] > 0:  # obj2 is in front of obj1
            if torch.abs(rel_vector[0]) > torch.abs(rel_vector[1]):
                return "right_of" if rel_vector[0] > 0 else "left_of"
            else:
                return "above" if rel_vector[1] > 0 else "below"
        else:  # obj2 is behind obj1
            return "behind"

class PointNet3D(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        # PointNet architecture
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, point_cloud):
        """Process point cloud through PointNet"""
        # Input: (batch_size, num_points, 3)
        batch_size, num_points, _ = point_cloud.shape

        # Apply MLP to each point
        point_features = self.mlp1(point_cloud.view(-1, 3))
        point_features = point_features.view(batch_size, num_points, -1)

        # Global feature aggregation (max pooling)
        global_features = torch.max(point_features, dim=1)[0]

        # Final feature extraction
        final_features = self.mlp2(global_features)

        return final_features

class VoxelEncoder(nn.Module):
    def __init__(self, grid_size=32):
        super().__init__()
        self.grid_size = grid_size

        # Voxel-based 3D CNN
        self.cnn3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(128, 512)
        )

    def forward(self, point_cloud):
        """Process point cloud through voxel encoding"""
        # Convert point cloud to voxel grid
        voxel_grid = self.point_cloud_to_voxel_grid(point_cloud)

        # Apply 3D CNN
        features = self.cnn3d(voxel_grid)

        return features

    def point_cloud_to_voxel_grid(self, point_cloud):
        """Convert point cloud to voxel grid representation"""
        # Normalize point cloud to grid bounds
        min_vals = torch.min(point_cloud, dim=1, keepdim=True)[0]
        max_vals = torch.max(point_cloud, dim=1, keepdim=True)[0]
        normalized_points = (point_cloud - min_vals) / (max_vals - min_vals + 1e-6)

        # Scale to voxel grid size
        scaled_points = normalized_points * (self.grid_size - 1)

        # Quantize to voxel indices
        voxel_indices = torch.floor(scaled_points).long()

        # Create voxel grid
        batch_size = point_cloud.shape[0]
        voxel_grid = torch.zeros((batch_size, 1, self.grid_size, self.grid_size, self.grid_size))

        # Fill voxel grid
        for b in range(batch_size):
            valid_mask = (voxel_indices[b] >= 0) & (voxel_indices[b] < self.grid_size)
            valid_mask = torch.all(valid_mask, dim=1)
            valid_indices = voxel_indices[b][valid_mask]

            # Fill occupied voxels
            voxel_grid[b, 0, valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1.0

        return voxel_grid
```

## Decision Making Systems

### Hierarchical Decision Making
```python
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import heapq

class HierarchicalDecisionMaker(nn.Module):
    def __init__(self, num_tasks=10, action_dim=7):
        super().__init__()

        # High-level task planner
        self.task_planner = TaskPlanner(num_tasks)

        # Mid-level motion planner
        self.motion_planner = MotionPlanner(action_dim)

        # Low-level action executor
        self.action_executor = ActionExecutor(action_dim)

        # Task-goal alignment
        self.task_alignment = TaskGoalAlignment()

        # Memory for context
        self.context_memory = ContextMemory()

    def forward(self, perception_input, goal_specification):
        """Make hierarchical decisions based on perception and goals"""
        # Update context memory
        self.context_memory.update(perception_input)

        # High-level task planning
        task_plan = self.task_planner(plan_context={
            'perception': perception_input,
            'goal': goal_specification,
            'memory': self.context_memory.get_recent_context()
        })

        # Mid-level motion planning
        motion_plan = self.motion_planner(plan_context={
            'task_plan': task_plan,
            'perception': perception_input,
            'obstacles': perception_input.get('obstacles', []),
            'targets': perception_input.get('targets', [])
        })

        # Low-level action execution
        action = self.action_executor.execute_context={
            'motion_plan': motion_plan,
            'perception': perception_input,
            'task_plan': task_plan
        })

        return {
            'task_plan': task_plan,
            'motion_plan': motion_plan,
            'action': action,
            'confidence': self.assess_confidence(perception_input, action)
        }

    def assess_confidence(self, perception_input, action):
        """Assess confidence in the decision"""
        # Evaluate based on sensor quality, obstacle density, etc.
        confidence = 1.0

        # Reduce confidence if sensor data is poor
        if perception_input.get('sensor_quality', 1.0) < 0.5:
            confidence *= 0.8

        # Reduce confidence if many obstacles nearby
        obstacles = perception_input.get('obstacles', [])
        if len(obstacles) > 5:
            confidence *= 0.7

        # Reduce confidence if action is complex
        action_complexity = torch.norm(action).item()
        if action_complexity > 0.8:
            confidence *= 0.9

        return confidence

class TaskPlanner(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks

        # Task selection network
        self.task_selector = nn.Sequential(
            nn.Linear(512, 256),  # Input features from perception
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_tasks),
            nn.Softmax(dim=-1)
        )

        # Task dependency graph
        self.task_dependencies = self.initialize_task_dependencies()

    def forward(self, plan_context):
        """Plan high-level tasks"""
        perception_features = plan_context['perception']['features']
        goal_features = plan_context['goal']['features']

        # Combine perception and goal features
        combined_features = torch.cat([perception_features, goal_features], dim=-1)

        # Select tasks
        task_probs = self.task_selector(combined_features)
        selected_task_idx = torch.argmax(task_probs, dim=-1)

        # Generate task sequence considering dependencies
        task_sequence = self.generate_task_sequence(selected_task_idx, plan_context['memory'])

        return {
            'selected_task': selected_task_idx,
            'task_probs': task_probs,
            'sequence': task_sequence,
            'dependencies': self.task_dependencies[selected_task_idx.item()]
        }

    def initialize_task_dependencies(self):
        """Initialize task dependency graph"""
        # Example dependencies: [walk, grasp, manipulate, avoid_obstacle, ...]
        dependencies = {
            0: [],  # walk
            1: [0],  # grasp requires walk to reach object
            2: [1],  # manipulate requires grasp
            3: [0],  # avoid_obstacle may interrupt other tasks
            # Add more dependencies as needed
        }
        return dependencies

    def generate_task_sequence(self, initial_task, memory):
        """Generate sequence of tasks considering dependencies"""
        # Use topological sort to respect dependencies
        visited = set()
        sequence = []

        def dfs(task_idx):
            if task_idx in visited:
                return
            visited.add(task_idx)

            # Visit dependencies first
            for dep in self.task_dependencies.get(task_idx, []):
                dfs(dep)

            sequence.append(task_idx)

        dfs(initial_task.item())
        return sequence

class MotionPlanner(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

        # Path planning network
        self.path_planner = nn.Sequential(
            nn.Linear(256, 512),  # Input: [start, goal, obstacles]
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # Obstacle avoidance
        self.obstacle_avoider = ObstacleAvoidanceNetwork()

    def forward(self, plan_context):
        """Plan motion trajectories"""
        task_plan = plan_context['task_plan']
        perception = plan_context['perception']
        obstacles = plan_context['obstacles']
        targets = plan_context['targets']

        # Plan path to target
        path = self.plan_path_to_target(targets, obstacles)

        # Generate motion commands
        motion_commands = self.generate_motion_commands(path, task_plan)

        return {
            'path': path,
            'commands': motion_commands,
            'avoidance_strategy': self.obstacle_avoider(obstacles)
        }

    def plan_path_to_target(self, targets, obstacles):
        """Plan path to target using A* or RRT"""
        # Simplified path planning
        if len(targets) > 0:
            target = targets[0]  # Use first target
            # In practice, this would use A* or RRT algorithm
            path = [target]  # Simplified
        else:
            path = []

        return path

    def generate_motion_commands(self, path, task_plan):
        """Generate motion commands for the path"""
        commands = []
        for waypoint in path:
            # Generate command to move to waypoint
            command = self.path_planner(torch.cat([waypoint, task_plan['selected_task'].float()]))
            commands.append(command)
        return commands

class ActionExecutor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

        # Action selection network
        self.action_selector = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

    def forward(self, execute_context):
        """Execute the planned action"""
        motion_plan = execute_context['motion_plan']
        perception = execute_context['perception']
        task_plan = execute_context['task_plan']

        # Select appropriate action based on context
        context_features = torch.cat([
            motion_plan['commands'][0] if motion_plan['commands'] else torch.zeros(self.action_dim),
            task_plan['selected_task'].float()
        ])

        action = self.action_selector(context_features)

        return action
```

## Reinforcement Learning for Decision Making

### Deep Reinforcement Learning for Robotics
```python
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class DDPGRobotAgent(nn.Module):
    def __init__(self, state_dim=512, action_dim=7, hidden_dim=256):
        super().__init__()

        # Actor network (policy network)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

        # Critic network (Q-function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q-value
        )

        # Target networks for stable learning
        self.actor_target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Replay buffer
        self.replay_buffer = deque(maxlen=100000)

        # Parameters
        self.action_dim = action_dim
        self.noise_std = 0.1
        self.noise_decay = 0.999

    def forward(self, state):
        """Get action for given state"""
        action = self.actor(state)
        return action

    def get_action_with_noise(self, state, add_noise=True):
        """Get action with optional exploration noise"""
        action = self.actor(state).detach().cpu().numpy()

        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
            self.noise_std *= self.noise_decay  # Decay noise over time

        return action

    def update(self, batch_size=64, gamma=0.99, tau=0.005):
        """Update networks using DDPG algorithm"""
        if len(self.replay_buffer) < batch_size:
            return

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Update critic
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(torch.cat([next_states, next_actions], dim=1))
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))

        current_q_values = self.critic(torch.cat([states, actions], dim=1))

        critic_loss = nn.MSELoss()(current_q_values, target_q_values.detach())

        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, predicted_actions], dim=1)).mean()

        # Optimize networks
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update target networks
        self.soft_update(self.actor, self.actor_target, tau)
        self.soft_update(self.critic, self.critic_target, tau)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

class PerceptualRLAgent(nn.Module):
    def __init__(self, vision_dim=512, proprioceptive_dim=20, action_dim=7):
        super().__init__()

        # Vision processing
        self.vision_processor = nn.Sequential(
            nn.Linear(vision_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Proprioceptive processing
        self.proprioceptive_processor = nn.Sequential(
            nn.Linear(proprioceptive_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Combined state processing
        combined_dim = 128 + 32
        self.state_processor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(128 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Target networks
        self.actor_target = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.critic_target = nn.Sequential(
            nn.Linear(128 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Replay buffer
        self.replay_buffer = deque(maxlen=50000)

    def forward(self, vision_features, proprioceptive_features):
        """Process perceptual inputs and generate action"""
        # Process vision
        vision_processed = self.vision_processor(vision_features)

        # Process proprioception
        proprio_processed = self.proprioceptive_processor(proprioceptive_features)

        # Combine features
        combined_features = torch.cat([vision_processed, proprio_processed], dim=-1)
        state_features = self.state_processor(combined_features)

        # Generate action
        action = self.actor(state_features)

        return action

    def get_action(self, vision_features, proprioceptive_features, add_noise=True):
        """Get action with optional exploration noise"""
        action = self.forward(vision_features, proprioceptive_features).detach().cpu().numpy()

        if add_noise:
            noise = np.random.normal(0, 0.1, size=len(action))
            action = np.clip(action + noise, -1, 1)

        return action
```

## Exercise: Implement AI Decision System

Create an AI system that:
1. Processes multimodal perception data (vision, audio, tactile)
2. Makes hierarchical decisions for complex tasks
3. Uses reinforcement learning for skill improvement
4. Integrates with the control system for execution

## Summary

The AI perception and decision-making systems form the cognitive core of the humanoid robot. By processing multiple sensory modalities and making intelligent decisions based on environmental understanding, these systems enable the robot to perform complex autonomous behaviors. The integration of computer vision, 3D scene understanding, hierarchical planning, and reinforcement learning creates a sophisticated AI system capable of navigating and interacting with the real world effectively.

---