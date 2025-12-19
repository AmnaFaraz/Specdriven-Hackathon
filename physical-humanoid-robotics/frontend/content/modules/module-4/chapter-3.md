---
id: module-4-chapter-3
title: "Action Generation and Execution in VLA Systems"
sidebar_label: "VLA Actions"
---

# Action Generation and Execution in VLA Systems

This chapter explores how Vision-Language-Action (VLA) systems generate and execute physical actions based on multimodal input, bridging the gap between perception and action in robotic systems.

## Action Space Representation

### Continuous Action Spaces
```python
import torch
import torch.nn as nn
import numpy as np

class ContinuousActionSpace(nn.Module):
    def __init__(self, action_dim=7, low_bounds=None, high_bounds=None):
        super().__init__()

        self.action_dim = action_dim

        # Define action bounds
        if low_bounds is None:
            self.low_bounds = torch.tensor([-1.0] * action_dim)
        else:
            self.low_bounds = torch.tensor(low_bounds)

        if high_bounds is None:
            self.high_bounds = torch.tensor([1.0] * action_dim)
        else:
            self.high_bounds = torch.tensor(high_bounds)

    def forward(self, action_logits):
        """Map action logits to continuous action space"""
        # Use tanh to map to [-1, 1] range
        raw_actions = torch.tanh(action_logits)

        # Scale to desired range
        scale = (self.high_bounds - self.low_bounds) / 2
        offset = (self.high_bounds + self.low_bounds) / 2

        scaled_actions = raw_actions * scale + offset

        return scaled_actions

    def sample_action(self, mean, std):
        """Sample action from Gaussian distribution"""
        noise = torch.randn_like(mean) * std
        action = mean + noise
        return torch.clamp(action, self.low_bounds, self.high_bounds)

class RobotActionGenerator(nn.Module):
    def __init__(self, vision_dim=512, text_dim=512, action_dim=7):
        super().__init__()

        # Fusion layer to combine vision and text features
        self.fusion_layer = nn.Sequential(
            nn.Linear(vision_dim + text_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Action generation layers
        self.action_mean = nn.Linear(256, action_dim)
        self.action_std = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softplus()  # Ensure positive standard deviation
        )

        # Action space
        self.action_space = ContinuousActionSpace(action_dim=action_dim)

    def forward(self, vision_features, text_features):
        """Generate actions from vision and text features"""
        # Combine vision and text features
        combined_features = torch.cat([vision_features, text_features], dim=-1)

        # Pass through fusion network
        fused_features = self.fusion_layer(combined_features)

        # Generate action parameters
        action_mean = self.action_mean(fused_features)
        action_std = self.action_std(fused_features)

        # Sample action from distribution
        action = self.action_space.sample_action(action_mean, action_std)

        return {
            'action': action,
            'mean': action_mean,
            'std': action_std,
            'features': fused_features
        }
```

### Discrete Action Spaces
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteActionSpace(nn.Module):
    def __init__(self, num_actions=20):
        super().__init__()
        self.num_actions = num_actions

    def forward(self, action_logits):
        """Convert action logits to discrete action"""
        # Apply softmax to get action probabilities
        action_probs = F.softmax(action_logits, dim=-1)

        # Sample action based on probabilities
        action_idx = torch.multinomial(action_probs, 1)

        return action_idx, action_probs

class DiscreteActionGenerator(nn.Module):
    def __init__(self, vision_dim=512, text_dim=512, num_actions=20):
        super().__init__()

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(vision_dim + text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Action prediction head
        self.action_head = nn.Linear(128, num_actions)

        # Action space
        self.action_space = DiscreteActionSpace(num_actions=num_actions)

    def forward(self, vision_features, text_features):
        """Generate discrete action from vision and text features"""
        # Combine features
        combined_features = torch.cat([vision_features, text_features], dim=-1)

        # Fuse features
        fused_features = self.fusion_layer(combined_features)

        # Predict action logits
        action_logits = self.action_head(fused_features)

        # Get discrete action
        action_idx, action_probs = self.action_space(action_logits)

        return {
            'action_idx': action_idx,
            'action_probs': action_probs,
            'logits': action_logits
        }
```

## Hierarchical Action Planning

### Hierarchical Action Network
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalActionPlanner(nn.Module):
    def __init__(self, vision_dim=512, text_dim=512, high_level_actions=10, low_level_actions=50):
        super().__init__()

        # High-level planner (goal-oriented)
        self.high_level_planner = nn.Sequential(
            nn.Linear(vision_dim + text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, high_level_actions)
        )

        # Low-level executor (motion primitive)
        self.low_level_executor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vision_dim + text_dim + high_level_actions, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, low_level_actions)
            ) for _ in range(high_level_actions)
        ])

        self.high_level_actions = high_level_actions
        self.low_level_actions = low_level_actions

    def forward(self, vision_features, text_features):
        """Generate hierarchical action plan"""
        # Combine vision and text
        combined_features = torch.cat([vision_features, text_features], dim=-1)

        # High-level planning
        high_level_logits = self.high_level_planner(combined_features)
        high_level_probs = F.softmax(high_level_logits, dim=-1)
        high_level_action = torch.argmax(high_level_probs, dim=-1)

        # Low-level execution based on high-level plan
        # One-hot encode high-level action
        high_action_onehot = F.one_hot(high_level_action, num_classes=self.high_level_actions).float()

        # Concatenate with vision and text features
        low_level_input = torch.cat([combined_features, high_action_onehot], dim=-1)

        # Select appropriate low-level network
        low_level_logits = []
        for i in range(self.high_level_actions):
            # Compute logits for each low-level network
            logits_i = self.low_level_executor[i](low_level_input)
            low_level_logits.append(logits_i)

        # Stack and select based on high-level action
        low_level_logits = torch.stack(low_level_logits, dim=1)  # [batch, high_actions, low_actions]

        # Select logits for the chosen high-level action
        batch_indices = torch.arange(len(high_level_action))
        selected_logits = low_level_logits[batch_indices, high_level_action]

        # Get low-level action
        low_level_probs = F.softmax(selected_logits, dim=-1)
        low_level_action = torch.argmax(low_level_probs, dim=-1)

        return {
            'high_level_action': high_level_action,
            'low_level_action': low_level_action,
            'high_level_probs': high_level_probs,
            'low_level_probs': low_level_probs
        }

    def get_action_sequence(self, vision_features, text_features, sequence_length=5):
        """Generate action sequence for a task"""
        actions = []

        for t in range(sequence_length):
            action_output = self.forward(vision_features, text_features)
            actions.append({
                'high': action_output['high_level_action'].item(),
                'low': action_output['low_level_action'].item(),
                'high_prob': action_output['high_level_probs'].max().item(),
                'low_prob': action_output['low_level_probs'].max().item()
            })

            # Update features based on expected outcome (simplified)
            # In practice, this would use the actual robot state after each action
            vision_features = vision_features * 0.95  # Simulate state transition

        return actions
```

## Task-Oriented Action Generation

### Task-Based Action Generator
```python
import torch
import torch.nn as nn
import numpy as np

class TaskOrientedActionGenerator(nn.Module):
    def __init__(self, vision_dim=512, text_dim=512, action_dim=7, num_tasks=10):
        super().__init__()

        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(vision_dim + text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_tasks)
        )

        # Task-specific action generators
        self.task_specific_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vision_dim + text_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh()  # Normalize to [-1, 1]
            ) for _ in range(num_tasks)
        ])

        # Task-specific value estimators
        self.task_value_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vision_dim + text_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(num_tasks)
        ])

        self.num_tasks = num_tasks
        self.action_dim = action_dim

    def forward(self, vision_features, text_features):
        """Generate task-oriented actions"""
        # Combine features
        combined_features = torch.cat([vision_features, text_features], dim=-1)

        # Classify task
        task_logits = self.task_classifier(combined_features)
        task_probs = F.softmax(task_logits, dim=-1)
        task_idx = torch.argmax(task_probs, dim=-1)

        # Generate action for the identified task
        actions = []
        values = []

        for i in range(self.num_tasks):
            # Generate action for each task
            task_action = self.task_specific_generators[i](combined_features)
            task_value = self.task_value_estimators[i](combined_features)

            actions.append(task_action)
            values.append(task_value)

        # Stack actions and values
        all_actions = torch.stack(actions, dim=1)  # [batch, num_tasks, action_dim]
        all_values = torch.stack(values, dim=1)    # [batch, num_tasks, 1]

        # Select action based on identified task
        batch_indices = torch.arange(len(task_idx))
        selected_actions = all_actions[batch_indices, task_idx]
        selected_values = all_values[batch_indices, task_idx]

        return {
            'task_idx': task_idx,
            'task_probs': task_probs,
            'action': selected_actions,
            'value': selected_values,
            'all_actions': all_actions,
            'all_values': all_values
        }

    def get_task_name(self, task_idx):
        """Get task name from index"""
        task_names = [
            "navigation", "grasping", "manipulation", "inspection",
            "transport", "assembly", "disassembly", "cleaning",
            "monitoring", "communication"
        ]
        return task_names[task_idx] if task_idx < len(task_names) else "unknown"
```

## Real-time Action Execution

### Action Execution Manager
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge
import torch
import time
import numpy as np

class ActionExecutionManager(Node):
    def __init__(self):
        super().__init__('action_execution_manager')

        # Initialize VLA components
        self.action_generator = TaskOrientedActionGenerator()
        self.action_generator.eval()

        # ROS 2 interfaces
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_rect_color', self.image_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, '/vla_command', self.command_callback, 10
        )

        # Action publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.pose_cmd_pub = self.create_publisher(Pose, '/pose_command', 10)

        # Processing components
        self.bridge = CvBridge()
        self.current_joint_state = None
        self.current_image = None
        self.pending_command = None

        # Action execution parameters
        self.execution_frequency = 10  # Hz
        self.action_timeout = 5.0  # seconds
        self.last_action_time = time.time()

    def joint_state_callback(self, msg):
        """Update joint state"""
        self.current_joint_state = msg

    def image_callback(self, msg):
        """Process incoming image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image

            # Process if we have a pending command
            if self.pending_command:
                self.execute_vla_action()
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process incoming command"""
        self.pending_command = msg.data

        # Process if we have current image
        if self.current_image:
            self.execute_vla_action()

    def execute_vla_action(self):
        """Execute VLA-generated action"""
        if not self.current_image or not self.pending_command:
            return

        try:
            # Preprocess inputs
            image_tensor = self.preprocess_image(self.current_image)
            text_tensor = self.preprocess_text(self.pending_command)

            # Generate action using VLA model
            with torch.no_grad():
                action_output = self.action_generator(image_tensor, text_tensor)

            # Extract action
            action = action_output['action'].cpu().numpy()[0]
            task_idx = action_output['task_idx'].cpu().numpy()[0]

            # Execute action based on task type
            self.execute_task_specific_action(action, task_idx)

            # Update execution time
            self.last_action_time = time.time()

            # Clear processed command
            self.pending_command = None

        except Exception as e:
            self.get_logger().error(f'Error executing VLA action: {e}')

    def preprocess_image(self, image):
        """Preprocess image for VLA model"""
        import cv2
        resized = cv2.resize(image, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def preprocess_text(self, text):
        """Preprocess text for VLA model"""
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        return inputs['input_ids']

    def execute_task_specific_action(self, action, task_idx):
        """Execute action based on task type"""
        if task_idx == 0:  # Navigation
            self.execute_navigation_action(action)
        elif task_idx == 1:  # Grasping
            self.execute_grasping_action(action)
        elif task_idx == 2:  # Manipulation
            self.execute_manipulation_action(action)
        else:  # Default action
            self.execute_default_action(action)

    def execute_navigation_action(self, action):
        """Execute navigation action"""
        cmd_vel = Twist()

        # Map action to navigation commands
        cmd_vel.linear.x = action[0]  # Forward/backward
        cmd_vel.angular.z = action[1]  # Turn left/right

        self.cmd_vel_pub.publish(cmd_vel)
        self.get_logger().info(f'Navigating: linear={cmd_vel.linear.x:.2f}, angular={cmd_vel.angular.z:.2f}')

    def execute_grasping_action(self, action):
        """Execute grasping action"""
        joint_cmd = JointState()
        joint_cmd.name = ['finger_joint_1', 'finger_joint_2']  # Example joint names
        joint_cmd.position = [action[0], action[1]]  # Finger positions

        self.joint_cmd_pub.publish(joint_cmd)
        self.get_logger().info(f'Grasping: fingers={[f"{pos:.2f}" for pos in joint_cmd.position]}')

    def execute_manipulation_action(self, action):
        """Execute manipulation action"""
        pose_cmd = Pose()

        # Map action to end-effector pose
        pose_cmd.position.x = action[0]
        pose_cmd.position.y = action[1]
        pose_cmd.position.z = action[2]
        pose_cmd.orientation.z = action[3]
        pose_cmd.orientation.w = action[4]

        self.pose_cmd_pub.publish(pose_cmd)
        self.get_logger().info(f'Manipulating: pos=({pose_cmd.position.x:.2f}, {pose_cmd.position.y:.2f}, {pose_cmd.position.z:.2f})')

    def execute_default_action(self, action):
        """Execute default action"""
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0] * 0.2  # Scale down for safety
        cmd_vel.angular.z = action[1] * 0.2

        self.cmd_vel_pub.publish(cmd_vel)
        self.get_logger().info(f'Default action: linear={cmd_vel.linear.x:.2f}, angular={cmd_vel.angular.z:.2f}')

    def check_action_timeout(self):
        """Check if action has timed out"""
        if time.time() - self.last_action_time > self.action_timeout:
            # Stop robot if action times out
            cmd_vel = Twist()
            self.cmd_vel_pub.publish(cmd_vel)
            self.get_logger().warn('Action timed out, stopping robot')
```

## Action Safety and Validation

### Action Safety Validator
```python
import torch
import torch.nn as nn
import numpy as np

class ActionSafetyValidator(nn.Module):
    def __init__(self, action_dim=7, safety_threshold=0.8):
        super().__init__()

        # Safety critic network
        self.safety_critic = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output safety probability [0, 1]
        )

        # Action constraint networks
        self.position_constraint = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.velocity_constraint = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.safety_threshold = safety_threshold

    def forward(self, action, current_state=None):
        """Validate action safety"""
        # Check action safety
        safety_score = self.safety_critic(action)

        # Check position constraints
        position_safe = self.position_constraint(action)

        # Check velocity constraints
        velocity_safe = self.velocity_constraint(action)

        # Combine safety scores
        overall_safety = (safety_score + position_safe + velocity_safe) / 3

        # Determine if action is safe
        is_safe = overall_safety > self.safety_threshold

        return {
            'is_safe': is_safe,
            'safety_score': overall_safety,
            'breakdown': {
                'safety': safety_score,
                'position': position_safe,
                'velocity': velocity_safe
            }
        }

    def safe_action_generation(self, unsafe_action, current_state=None):
        """Generate safe action from potentially unsafe one"""
        # Clamp action to safe ranges
        safe_action = torch.clamp(unsafe_action, -1.0, 1.0)

        # Further refine using safety validation
        validation = self.forward(safe_action, current_state)

        if not validation['is_safe']:
            # Reduce action magnitude until safe
            scale_factor = 0.9
            while not validation['is_safe'] and scale_factor > 0.1:
                scaled_action = safe_action * scale_factor
                validation = self.forward(scaled_action, current_state)
                scale_factor -= 0.1

            safe_action = scaled_action if validation['is_safe'] else torch.zeros_like(safe_action)

        return safe_action, validation
```

## Exercise: Implement Safe Action Execution System

Create a system that:
1. Generates actions from vision-language input
2. Validates actions for safety before execution
3. Implements hierarchical planning for complex tasks
4. Provides feedback on action execution success

## Summary

Action generation and execution in VLA systems represents the critical link between perception and physical interaction. The systems must not only generate appropriate actions based on multimodal input but also ensure these actions are safe, feasible, and aligned with the intended task. Hierarchical planning, safety validation, and real-time execution are essential components for creating robust and reliable robotic systems.

---