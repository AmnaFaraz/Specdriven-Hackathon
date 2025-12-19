---
id: module-3-chapter-5
title: "Isaac AI Brain: Learning and Adaptation"
sidebar_label: "Isaac Learning"
---

# Isaac AI Brain: Learning and Adaptation

This chapter explores how NVIDIA Isaac enables machine learning and adaptation capabilities that allow robots to learn from experience and adapt to new situations.

## Isaac Learning Framework

Isaac Learning includes:

- **Reinforcement Learning**: Training agents through interaction with environment
- **Imitation Learning**: Learning from human demonstrations
- **Transfer Learning**: Adapting pre-trained models to new tasks
- **Online Adaptation**: Real-time learning and adjustment

## Isaac Reinforcement Learning

### Deep Q-Network for Robot Control
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class IsaacDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(IsaacDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class IsaacRLAgent(Node):
    def __init__(self):
        super().__init__('isaac_rl_agent')

        # Subscriptions
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_rect_color', self.image_callback, 10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reward_pub = self.create_publisher(Float32, '/rl_reward', 10)

        # RL parameters
        self.state_size = 360  # Laser scan points
        self.action_size = 9   # Discrete actions
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95

        # Neural networks
        self.q_network = IsaacDQN(self.state_size, self.action_size)
        self.target_network = IsaacDQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Training parameters
        self.batch_size = 32
        self.update_target_freq = 100
        self.step_count = 0

        # Robot state
        self.current_scan = None
        self.current_image = None
        self.previous_action = 0
        self.previous_reward = 0.0

    def laser_callback(self, msg):
        """Process laser scan for state representation"""
        self.current_scan = np.array(msg.ranges)
        # Replace invalid ranges with maximum range
        self.current_scan[np.isnan(self.current_scan)] = msg.range_max
        self.current_scan[np.isinf(self.current_scan)] = msg.range_max

        # Take action based on current state
        if self.current_scan is not None:
            action = self.act(self.current_scan)
            self.execute_action(action)

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def calculate_reward(self, action, scan_data):
        """Calculate reward based on robot state and action"""
        reward = 0.0

        # Avoid obstacles
        min_distance = np.min(scan_data)
        if min_distance < 0.5:
            reward -= 1.0  # Penalty for being too close to obstacles

        # Move forward reward
        if action == 4:  # Forward action
            reward += 0.1

        # Progress reward (simplified)
        reward += 0.01

        return reward

    def execute_action(self, action):
        """Execute action and calculate reward"""
        cmd_vel = Twist()

        # Map discrete action to velocity commands
        action_map = {
            0: (-0.2, -0.5),  # Sharp left
            1: (-0.1, -0.2),  # Left
            2: (0.0, -0.1),   # Slight left
            3: (-0.2, 0.0),   # Back left
            4: (0.5, 0.0),    # Forward
            5: (0.0, 0.1),    # Slight right
            6: (0.2, 0.0),    # Back right
            7: (0.1, 0.2),    # Right
            8: (0.2, 0.5)     # Sharp right
        }

        if action in action_map:
            cmd_vel.linear.x, cmd_vel.angular.z = action_map[action]

        self.cmd_vel_pub.publish(cmd_vel)

        # Calculate and store reward
        if self.current_scan is not None:
            reward = self.calculate_reward(action, self.current_scan)
            self.reward_pub.publish(Float32(data=reward))

            # Store experience for training
            if hasattr(self, 'previous_scan'):
                self.remember(self.previous_scan, self.previous_action,
                            self.previous_reward, self.current_scan, False)

            self.previous_action = action
            self.previous_reward = reward
            self.previous_scan = self.current_scan.copy()

        # Train network periodically
        if self.step_count % 10 == 0:
            self.replay()

        # Update target network periodically
        if self.step_count % self.update_target_freq == 0:
            self.update_target_network()

        self.step_count += 1
```

## Isaac Imitation Learning

### Learning from Demonstrations
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import torch
import torch.nn as nn
import numpy as np

class IsaacImitationNetwork(nn.Module):
    def __init__(self, image_shape, laser_size, action_size):
        super(IsaacImitationNetwork, self).__init__()

        # CNN for image processing
        self.conv1 = nn.Conv2d(image_shape[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate CNN output size
        conv_out_size = self.calculate_conv_output_size(image_shape)

        # FC layers combining image and laser features
        self.fc1 = nn.Linear(conv_out_size + laser_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.action_head = nn.Linear(256, action_size)
        self.value_head = nn.Linear(256, 1)

    def calculate_conv_output_size(self, image_shape):
        """Calculate the output size of convolutional layers"""
        h, w, c = image_shape
        h = (h - 8) // 4 + 1
        h = (h - 4) // 2 + 1
        h = (h - 3) // 1 + 1
        w = (w - 8) // 4 + 1
        w = (w - 4) // 2 + 1
        w = (w - 3) // 1 + 1
        return h * w * 64

    def forward(self, image, laser):
        # Process image through CNN
        x = torch.relu(self.conv1(image))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten

        # Concatenate with laser data
        combined = torch.cat([x, laser], dim=1)

        # Process through FC layers
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))

        # Output action and value
        action = self.action_head(x)
        value = self.value_head(x)

        return action, value

class IsaacImitationLearner(Node):
    def __init__(self):
        super().__init__('isaac_imitation_learner')

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/image_rect_color', self.image_callback, 10
        )
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )
        self.expert_cmd_sub = self.create_subscription(
            Twist, '/expert_cmd_vel', self.expert_command_callback, 10
        )
        self.demo_start_sub = self.create_subscription(
            Bool, '/start_demonstration', self.start_demonstration_callback, 10
        )

        # Publishers
        self.agent_cmd_pub = self.create_publisher(Twist, '/agent_cmd_vel', 10)

        # Network and training
        self.network = IsaacImitationNetwork(
            image_shape=(480, 640, 3),  # H, W, C
            laser_size=360,
            action_size=2  # linear.x, angular.z
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

        # Data collection
        self.demonstration_data = []
        self.is_collecting = False
        self.current_image = None
        self.current_laser = None
        self.current_expert_cmd = None

    def image_callback(self, msg):
        """Process image data"""
        # Convert ROS Image to PyTorch tensor
        image_np = self.ros_image_to_numpy(msg)
        self.current_image = torch.FloatTensor(image_np).permute(2, 0, 1).unsqueeze(0)  # CHW format

    def laser_callback(self, msg):
        """Process laser scan data"""
        laser_data = np.array(msg.ranges)
        laser_data[np.isnan(laser_data)] = msg.range_max
        laser_data[np.isinf(laser_data)] = msg.range_max
        self.current_laser = torch.FloatTensor(laser_data).unsqueeze(0)

    def expert_command_callback(self, msg):
        """Store expert demonstration commands"""
        self.current_expert_cmd = torch.FloatTensor([msg.linear.x, msg.angular.z]).unsqueeze(0)

    def start_demonstration_callback(self, msg):
        """Start/stop demonstration collection"""
        self.is_collecting = msg.data
        if not self.is_collecting:
            # Train on collected data
            self.train_on_demonstrations()

    def collect_demonstration_step(self):
        """Collect one step of demonstration"""
        if (self.is_collecting and
            self.current_image is not None and
            self.current_laser is not None and
            self.current_expert_cmd is not None):

            self.demonstration_data.append({
                'image': self.current_image.clone(),
                'laser': self.current_laser.clone(),
                'command': self.current_expert_cmd.clone()
            })

    def train_on_demonstrations(self):
        """Train network on collected demonstrations"""
        if len(self.demonstration_data) == 0:
            return

        for epoch in range(10):  # Train for 10 epochs
            total_loss = 0
            for data in self.demonstration_data:
                # Forward pass
                predicted_action, _ = self.network(data['image'], data['laser'])

                # Calculate loss (MSE between predicted and expert action)
                loss = nn.MSELoss()(predicted_action, data['command'])

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            self.get_logger().info(f'Epoch {epoch}, Average Loss: {total_loss/len(self.demonstration_data)}')

        # Clear demonstration data after training
        self.demonstration_data = []

    def execute_policy(self):
        """Execute learned policy"""
        if (self.current_image is not None and
            self.current_laser is not None):

            with torch.no_grad():
                action, _ = self.network(self.current_image, self.current_laser)

            cmd_vel = Twist()
            cmd_vel.linear.x = action[0, 0].item()
            cmd_vel.angular.z = action[0, 1].item()

            self.agent_cmd_pub.publish(cmd_vel)
```

## Isaac Transfer Learning

### Adapting Pre-trained Models
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import torch
import torchvision.models as models
import torch.nn as nn

class IsaacTransferLearner(Node):
    def __init__(self):
        super().__init__('isaac_transfer_learner')

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/image_rect_color', self.image_callback, 10
        )

        # Publishers
        self.prediction_pub = self.create_publisher(String, '/transfer_prediction', 10)

        # Load pre-trained model
        self.pretrained_model = models.resnet18(pretrained=True)

        # Replace final layer for specific task (e.g., object classification in robotics)
        num_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(num_features, 10)  # 10 classes for robotics objects

        # Freeze early layers (transfer learning approach)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Only train the final layer
        for param in self.pretrained_model.fc.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.Adam(self.pretrained_model.fc.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        self.current_image = None
        self.is_training = False

    def image_callback(self, msg):
        """Process image and make predictions"""
        image_tensor = self.ros_image_to_tensor(msg)

        with torch.no_grad():
            output = self.pretrained_model(image_tensor)
            predicted_class = torch.argmax(output, dim=1)

        # Publish prediction
        pred_msg = String()
        pred_msg.data = f"Class {predicted_class.item()}"
        self.prediction_pub.publish(pred_msg)

    def fine_tune_model(self, new_dataset):
        """Fine-tune the model on new robotic-specific data"""
        self.pretrained_model.train()

        for epoch in range(5):  # Few epochs for fine-tuning
            total_loss = 0
            for batch_idx, (data, target) in enumerate(new_dataset):
                self.optimizer.zero_grad()
                output = self.pretrained_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.get_logger().info(f'Fine-tuning Epoch {epoch}, Loss: {total_loss/len(new_dataset)}')

        # Now unfreeze more layers for further training if needed
        for param in list(self.pretrained_model.parameters())[-5:]:  # Unfreeze last 5 layers
            param.requires_grad = True
```

## Isaac Online Adaptation

### Real-time Learning and Adjustment
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

class IsaacOnlineAdapter(Node):
    def __init__(self):
        super().__init__('isaac_online_adapter')

        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.command_sub = self.create_subscription(
            Twist, '/cmd_vel', self.command_callback, 10
        )

        # Publishers
        self.adapted_cmd_pub = self.create_publisher(Twist, '/adapted_cmd_vel', 10)
        self.adaptation_params_pub = self.create_publisher(
            Float32MultiArray, '/adaptation_params', 10
        )

        # Online learning components
        self.feature_scaler = StandardScaler()
        self.adaptation_model = SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1
        )

        # Robot state tracking
        self.joint_positions = np.zeros(18)
        self.joint_velocities = np.zeros(18)
        self.imu_data = np.zeros(6)  # [angular_vel, linear_acc]
        self.command_history = []
        self.performance_history = []

        # Adaptation parameters
        self.adaptation_enabled = True
        self.performance_threshold = 0.8
        self.adaptation_rate = 0.1

    def joint_state_callback(self, msg):
        """Update joint state information"""
        for i, name in enumerate(msg.name):
            # Update joint positions and velocities
            if i < len(self.joint_positions):
                self.joint_positions[i] = msg.position[i]
            if i < len(self.joint_velocities):
                self.joint_velocities[i] = msg.velocity[i]

    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_data[:3] = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]
        self.imu_data[3:] = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ]

    def command_callback(self, msg):
        """Process command and adapt if needed"""
        # Create feature vector from current state
        features = self.create_feature_vector(msg)

        # Apply adaptation if enabled
        if self.adaptation_enabled:
            adapted_cmd = self.apply_adaptation(msg, features)
        else:
            adapted_cmd = msg

        # Publish adapted command
        self.adapted_cmd_pub.publish(adapted_cmd)

        # Update adaptation model with new data
        self.update_adaptation_model(features, msg, adapted_cmd)

    def create_feature_vector(self, command):
        """Create feature vector for adaptation"""
        features = np.concatenate([
            self.joint_positions[:6],      # First 6 joints for simplicity
            self.joint_velocities[:6],     # Velocities
            self.imu_data,                 # IMU data
            [command.linear.x, command.linear.y, command.linear.z],
            [command.angular.x, command.angular.y, command.angular.z]
        ])

        # Normalize features
        if len(self.command_history) > 10:  # Enough data for scaling
            features = self.feature_scaler.transform(features.reshape(1, -1)).flatten()
        else:
            self.feature_scaler.partial_fit(features.reshape(1, -1))

        return features

    def apply_adaptation(self, original_cmd, features):
        """Apply online adaptation to command"""
        # Predict adaptation offsets using the model
        if hasattr(self.adaptation_model, 'coef_'):
            adaptation_offset = self.adaptation_model.predict(features.reshape(1, -1))[0]

            # Apply adaptation to command
            adapted_cmd = Twist()
            adapted_cmd.linear.x = original_cmd.linear.x + adaptation_offset * 0.1
            adapted_cmd.angular.z = original_cmd.angular.z + adaptation_offset * 0.05

            # Constrain to safe limits
            adapted_cmd.linear.x = np.clip(adapted_cmd.linear.x, -1.0, 1.0)
            adapted_cmd.angular.z = np.clip(adapted_cmd.angular.z, -1.0, 1.0)

            return adapted_cmd
        else:
            return original_cmd

    def update_adaptation_model(self, features, original_cmd, adapted_cmd):
        """Update adaptation model with new experience"""
        # Calculate performance based on command execution
        performance = self.calculate_performance(original_cmd, adapted_cmd)

        # Store for history
        self.command_history.append((features, original_cmd))
        self.performance_history.append(performance)

        # Keep only recent history
        if len(self.command_history) > 1000:
            self.command_history.pop(0)
            self.performance_history.pop(0)

        # Update model if we have enough good examples
        if len(self.performance_history) > 10 and np.mean(self.performance_history[-10:]) > self.performance_threshold:
            # Use the most recent data to update the model
            recent_features = np.array([x[0] for x in self.command_history[-10:]])
            recent_performance = np.array(self.performance_history[-10:])

            # Partial fit for online learning
            self.adaptation_model.partial_fit(recent_features, recent_performance)

    def calculate_performance(self, original_cmd, adapted_cmd):
        """Calculate performance metric for adaptation"""
        # This would typically involve comparing expected vs actual results
        # For simplicity, using a placeholder based on IMU stability
        stability_score = 1.0 - min(0.5, np.linalg.norm(self.imu_data[3:])/10.0)  # Linear acceleration
        return stability_score
```

## Exercise: Implement Adaptive Control System

Create a system that:
1. Learns from human demonstrations
2. Adapts its behavior based on environmental feedback
3. Uses transfer learning to apply knowledge to new tasks
4. Continuously improves through online learning

## Summary

Isaac provides powerful learning and adaptation capabilities that enable robots to improve their performance over time. Through reinforcement learning, imitation learning, transfer learning, and online adaptation, robots can become more capable and efficient as they gain experience. These AI brain capabilities are essential for creating truly autonomous and adaptable robotic systems.

---