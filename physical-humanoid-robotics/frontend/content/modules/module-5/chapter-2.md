---
id: module-5-chapter-2
title: "Humanoid Robot Control Systems"
sidebar_label: "Control Systems"
---

# Humanoid Robot Control Systems

This chapter delves into the control systems that enable precise and stable movement of the humanoid robot, including whole-body control, balance maintenance, and coordinated motion planning.

## Control Architecture

The humanoid robot employs a hierarchical control architecture with multiple layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Control Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│  Behavior Layer                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  High-Level Behaviors: Walking, Standing, Grasping, etc.    │ │
│  │  Task Planning and Sequencing                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Motion Planning Layer                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Trajectory Generation: Joint Space, Cartesian Space       │ │
│  │  Inverse Kinematics, Dynamic Walking Patterns               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Feedback Control Layer                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  PID Controllers, Impedance Control, Balance Control       │ │
│  │  Real-Time Joint Control                                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Hardware Interface Layer                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Motor Drivers, Actuator Control, Sensor Fusion            │ │
│  │  Low-Level Hardware Abstraction                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Whole-Body Control Framework

### Inverse Kinematics Solver
```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import casadi as ca

class InverseKinematicsSolver:
    def __init__(self, robot_model):
        self.model = robot_model
        self.dof = len(robot_model.joint_names)

        # Define symbolic variables
        self.q = ca.MX.sym('q', self.dof)  # Joint angles
        self.x_des = ca.MX.sym('x_des', 6)  # Desired pose (position + orientation)

        # Forward kinematics functions
        self.end_effector_fk = self.define_forward_kinematics()

        # Optimization variables
        self.opti = ca.Opti()
        self.opt_q = self.opti.variable(self.dof)

    def define_forward_kinematics(self):
        """Define forward kinematics for the robot"""
        # This would use DH parameters or other kinematic model
        # For simplicity, using a basic implementation
        def fk(joint_angles):
            # Calculate end-effector pose from joint angles
            # This is a simplified example
            ee_pose = np.zeros(6)  # [x, y, z, rx, ry, rz]

            # Calculate based on robot model
            # This would be more complex in practice
            for i, angle in enumerate(joint_angles):
                # Simplified forward kinematics calculation
                pass

            return ee_pose

        return fk

    def solve_ik(self, target_pose, current_joints, weights=None):
        """Solve inverse kinematics problem"""
        if weights is None:
            weights = np.ones(self.dof)

        # Set up optimization problem
        self.opti.set_initial(self.opt_q, current_joints)

        # Objective: minimize joint deviations
        objective = ca.mtimes(
            (self.opt_q - current_joints).T,
            ca.diag(weights),
            (self.opt_q - current_joints)
        )

        # Add constraints
        ee_pose = self.forward_kinematics(self.opt_q)
        self.opti.subject_to(ca.norm_2(ee_pose[:3] - target_pose[:3]) <= 0.01)  # Position tolerance
        self.opti.subject_to(ca.norm_2(ee_pose[3:] - target_pose[3:]) <= 0.1)  # Orientation tolerance

        # Joint limits
        for i, (name, limits) in enumerate(self.model.joint_limits.items()):
            self.opti.subject_to(self.opt_q[i] >= limits['min'])
            self.opti.subject_to(self.opt_q[i] <= limits['max'])

        # Solve optimization
        self.opti.minimize(objective)

        # Set solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0
        }
        self.opti.solver('ipopt', opts)

        try:
            sol = self.opti.solve()
            return sol.value(self.opt_q)
        except:
            # Return current joints if no solution found
            return current_joints

    def forward_kinematics(self, joint_angles):
        """Calculate forward kinematics"""
        # Simplified implementation
        # In practice, this would use the robot's DH parameters or URDF
        pose = np.zeros(6)

        # Calculate transformation matrices for each joint
        T = np.eye(4)  # Homogeneous transformation

        for i, angle in enumerate(joint_angles):
            # Calculate joint transformation
            # This is a simplified example
            pass

        return pose
```

### Whole-Body Controller
```python
import numpy as np
from scipy.linalg import block_diag

class WholeBodyController:
    def __init__(self, robot_model):
        self.model = robot_model
        self.mass_matrix = None
        self.coriolis_matrix = None
        self.gravity_vector = None

        # Task priorities
        self.tasks = {
            'balance': {'priority': 1, 'weight': 1.0},
            'end_effector': {'priority': 2, 'weight': 0.8},
            'posture': {'priority': 3, 'weight': 0.5},
            'obstacle_avoidance': {'priority': 4, 'weight': 0.3}
        }

        # Initialize controllers
        self.balance_controller = BalanceController(robot_model)
        self.impedance_controllers = {}
        for joint_name in robot_model.joint_names:
            self.impedance_controllers[joint_name] = ImpedanceController(
                stiffness=100, damping=10
            )

    def compute_control(self, desired_states, current_states, external_forces=None):
        """Compute whole-body control torques"""
        # Update robot dynamics
        self.update_dynamics(current_states['joint_positions'],
                           current_states['joint_velocities'])

        # Compute task-space commands
        task_commands = self.compute_task_commands(desired_states, current_states)

        # Prioritize tasks using weighted least squares
        joint_torques = self.prioritize_tasks(task_commands, current_states)

        # Add gravity compensation
        joint_torques += self.gravity_vector

        # Add Coriolis and centrifugal terms
        joint_torques += np.dot(self.coriolis_matrix, current_states['joint_velocities'])

        return joint_torques

    def compute_task_commands(self, desired_states, current_states):
        """Compute task-space commands for each priority"""
        commands = {}

        # Balance task
        if 'balance' in desired_states:
            commands['balance'] = self.balance_controller.compute_balance_torque(
                desired_states['balance'], current_states
            )

        # End-effector task
        if 'end_effector' in desired_states:
            commands['end_effector'] = self.compute_end_effector_command(
                desired_states['end_effector'], current_states
            )

        # Posture task
        if 'posture' in desired_states:
            commands['posture'] = self.compute_posture_command(
                desired_states['posture'], current_states
            )

        return commands

    def prioritize_tasks(self, task_commands, current_states):
        """Prioritize tasks using weighted least squares"""
        # This implements the task-priority framework
        # Higher priority tasks are satisfied first

        # Construct task Jacobians
        J_balance = self.compute_balance_jacobian(current_states)
        J_ee = self.compute_end_effector_jacobian(current_states)
        J_posture = self.compute_posture_jacobian(current_states)

        # Construct stacked Jacobian matrix
        J_stack = np.vstack([
            self.tasks['balance']['weight'] * J_balance,
            self.tasks['end_effector']['weight'] * J_ee,
            self.tasks['posture']['weight'] * J_posture
        ])

        # Compute nullspace projection for lower priority tasks
        # J_pinv = J_stack^T * (J_stack * J_stack^T)^(-1)
        J_pinv = np.linalg.pinv(J_stack)

        # Compute desired accelerations for each task
        ddq_balance = task_commands['balance']
        ddq_ee = task_commands['end_effector']
        ddq_posture = task_commands['posture']

        ddq_stack = np.hstack([
            self.tasks['balance']['weight'] * ddq_balance,
            self.tasks['end_effector']['weight'] * ddq_ee,
            self.tasks['posture']['weight'] * ddq_posture
        ])

        # Compute joint accelerations
        ddq = np.dot(J_pinv, ddq_stack)

        # Compute joint torques using inverse dynamics
        tau = np.dot(self.mass_matrix, ddq)

        return tau

    def update_dynamics(self, q, dq):
        """Update robot dynamics matrices"""
        # Compute mass matrix M(q)
        self.mass_matrix = self.compute_mass_matrix(q)

        # Compute Coriolis matrix C(q,dq)
        self.coriolis_matrix = self.compute_coriolis_matrix(q, dq)

        # Compute gravity vector g(q)
        self.gravity_vector = self.compute_gravity_vector(q)

    def compute_mass_matrix(self, q):
        """Compute mass matrix using composite rigid body algorithm"""
        # Simplified implementation
        # In practice, this would use recursive Newton-Euler or other methods
        return np.eye(len(q)) * 1.0  # Simplified diagonal matrix

    def compute_coriolis_matrix(self, q, dq):
        """Compute Coriolis and centrifugal matrix"""
        # Simplified implementation
        return np.zeros((len(q), len(q)))

    def compute_gravity_vector(self, q):
        """Compute gravity vector"""
        # Simplified implementation
        g = 9.81
        return np.zeros(len(q))
```

## Balance Control Systems

### Center of Mass (CoM) Control
```python
import numpy as np
from scipy import signal
from collections import deque

class CoMController:
    def __init__(self, robot_model, control_frequency=100):
        self.model = robot_model
        self.dt = 1.0 / control_frequency
        self.control_frequency = control_frequency

        # CoM tracking controller (PID)
        self.com_pid = {
            'x': {'kp': 1000.0, 'ki': 100.0, 'kd': 100.0, 'integral': 0.0, 'prev_error': 0.0},
            'y': {'kp': 1000.0, 'ki': 100.0, 'kd': 100.0, 'integral': 0.0, 'prev_error': 0.0},
            'z': {'kp': 500.0, 'ki': 50.0, 'kd': 50.0, 'integral': 0.0, 'prev_error': 0.0}
        }

        # Support polygon (convex hull of feet contact points)
        self.support_polygon = None

        # State estimation
        self.com_position = np.array([0.0, 0.0, 0.8])  # Initial CoM position
        self.com_velocity = np.array([0.0, 0.0, 0.0])
        self.com_acceleration = np.array([0.0, 0.0, 0.0])

        # IMU data
        self.imu_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion
        self.imu_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.imu_linear_acceleration = np.array([0.0, 0.0, 9.81])

        # Foot positions (relative to robot base)
        self.left_foot_pos = np.array([0.0, 0.1, 0.0])  # Relative to robot base
        self.right_foot_pos = np.array([0.0, -0.1, 0.0])

        # Filters for sensor data
        self.com_filter = LowPassFilter(cutoff_freq=10.0, dt=self.dt)

    def update_sensor_data(self, imu_data, joint_positions, joint_velocities):
        """Update sensor data for balance control"""
        # Update IMU data
        self.imu_orientation = imu_data['orientation']
        self.imu_angular_velocity = imu_data['angular_velocity']
        self.imu_linear_acceleration = imu_data['linear_acceleration']

        # Update CoM estimation
        self.update_com_estimation(joint_positions, joint_velocities)

        # Update support polygon
        self.update_support_polygon()

    def update_com_estimation(self, joint_positions, joint_velocities):
        """Estimate CoM position and velocity"""
        # This would use the robot's kinematic model and joint positions
        # For now, using simplified estimation

        # Calculate CoM based on joint positions and masses
        # This is a simplified example
        total_mass = sum(self.model.link_masses)
        com_pos = np.zeros(3)

        for i, (joint_pos, mass) in enumerate(zip(joint_positions, self.model.link_masses)):
            # Calculate contribution of each link to CoM
            link_com = self.calculate_link_com(i, joint_pos)
            com_pos += mass * link_com

        com_pos /= total_mass

        # Update CoM state with filtering
        self.com_position = self.com_filter.filter(com_pos)

        # Estimate velocity using finite difference
        if hasattr(self, '_prev_com_pos'):
            self.com_velocity = (self.com_position - self._prev_com_pos) / self.dt
        self._prev_com_pos = self.com_position.copy()

    def calculate_link_com(self, link_idx, joint_pos):
        """Calculate center of mass for a specific link"""
        # Simplified calculation
        # In practice, this would use the URDF/kinematic model
        return joint_pos

    def update_support_polygon(self):
        """Update support polygon based on foot contacts"""
        # Calculate support polygon from foot positions
        # This would depend on current stance (double support, single support)

        if self.is_double_support():
            # Both feet on ground
            self.support_polygon = self.calculate_double_support_polygon()
        else:
            # Single foot support
            support_foot = self.get_support_foot()
            self.support_polygon = self.calculate_single_support_polygon(support_foot)

    def is_double_support(self):
        """Check if in double support phase"""
        # This would be determined by gait phase or contact sensors
        return True  # Simplified assumption

    def get_support_foot(self):
        """Get which foot is in support"""
        # This would be determined by gait phase or contact sensors
        return 'left'  # Simplified assumption

    def calculate_double_support_polygon(self):
        """Calculate support polygon for double support"""
        # Create convex hull of both feet
        left_pos = self.left_foot_pos
        right_pos = self.right_foot_pos

        # Simple rectangle approximation
        min_x, max_x = min(left_pos[0], right_pos[0]), max(left_pos[0], right_pos[0])
        min_y, max_y = min(left_pos[1], right_pos[1]), max(left_pos[1], right_pos[1])

        return np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])

    def calculate_single_support_polygon(self, foot):
        """Calculate support polygon for single support"""
        # Use foot size to define support polygon
        foot_pos = self.left_foot_pos if foot == 'left' else self.right_foot_pos

        # Simple rectangular support area
        foot_size_x, foot_size_y = 0.1, 0.06  # Typical foot size

        return np.array([
            [foot_pos[0] - foot_size_x/2, foot_pos[1] - foot_size_y/2],
            [foot_pos[0] + foot_size_x/2, foot_pos[1] - foot_size_y/2],
            [foot_pos[0] + foot_size_x/2, foot_pos[1] + foot_size_y/2],
            [foot_pos[0] - foot_size_x/2, foot_pos[1] + foot_size_y/2]
        ])

    def compute_balance_control(self, target_com_position):
        """Compute balance control torques"""
        # Calculate CoM error
        com_error = target_com_position - self.com_position

        # Apply PID control for each axis
        balance_torques = np.zeros(len(self.model.joint_names))

        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            if axis_idx >= len(com_error):
                continue

            error = com_error[axis_idx]

            # PID calculations
            self.com_pid[axis_name]['integral'] += error * self.dt
            derivative = (error - self.com_pid[axis_name]['prev_error']) / self.dt

            # Saturation for integral term
            self.com_pid[axis_name]['integral'] = np.clip(
                self.com_pid[axis_name]['integral'], -1.0, 1.0
            )

            control_output = (
                self.com_pid[axis_name]['kp'] * error +
                self.com_pid[axis_name]['ki'] * self.com_pid[axis_name]['integral'] +
                self.com_pid[axis_name]['kd'] * derivative
            )

            self.com_pid[axis_name]['prev_error'] = error

            # Map CoM control to joint torques
            # This would use the CoM Jacobian in practice
            joint_torques = self.map_com_to_joints(control_output, axis_idx)
            balance_torques += joint_torques

        return balance_torques

    def map_com_to_joints(self, com_control, axis_idx):
        """Map CoM control to joint torques"""
        # This would use the CoM Jacobian in practice
        # For now, return a simplified mapping
        torques = np.zeros(len(self.model.joint_names))

        # Simple mapping: torso joints for x/y CoM control, hip joints for z
        if axis_idx == 0:  # x-axis (forward/back)
            torques[6:12] = com_control * 0.1  # hip joints
        elif axis_idx == 1:  # y-axis (lateral)
            torques[6:12] = com_control * 0.1  # hip joints
        elif axis_idx == 2:  # z-axis (vertical)
            torques[0:6] = com_control * 0.05  # ankle joints

        return torques
```

## Walking Control Systems

### Walking Pattern Generator
```python
import numpy as np
import math
from scipy import interpolate

class WalkingPatternGenerator:
    def __init__(self, robot_model):
        self.model = robot_model
        self.gait_params = {
            'step_length': 0.3,      # meters
            'step_width': 0.2,       # meters (distance between feet)
            'step_height': 0.05,     # meters (foot lift height)
            'step_duration': 1.0,    # seconds per step
            'stance_ratio': 0.6,     # portion of step in stance phase
            'double_support_ratio': 0.1  # portion of step in double support
        }

        # Current gait state
        self.current_phase = 0.0
        self.current_support_foot = 'left'
        self.step_count = 0

        # Trajectory buffers
        self.foot_trajectory = {'left': [], 'right': []}
        self.com_trajectory = []

    def set_walking_parameters(self, step_length=0.3, step_height=0.05, step_duration=1.0):
        """Set walking parameters"""
        self.gait_params['step_length'] = step_length
        self.gait_params['step_height'] = step_height
        self.gait_params['step_duration'] = step_duration

        # Adjust other parameters based on step length
        self.gait_params['step_width'] = min(0.3, step_length * 0.66)  # Proportional to step length

    def generate_step_trajectory(self, target_velocity, dt=0.01):
        """Generate walking trajectory for next step"""
        # Update gait phase
        self.current_phase += dt / self.gait_params['step_duration']

        if self.current_phase >= 1.0:
            # Complete step cycle
            self.current_phase = 0.0
            self.step_count += 1

            # Switch support foot
            self.current_support_foot = 'right' if self.current_support_foot == 'left' else 'left'

        # Generate trajectories for both feet
        left_foot_pos = self.generate_foot_trajectory('left', target_velocity)
        right_foot_pos = self.generate_foot_trajectory('right', target_velocity)

        # Generate CoM trajectory for balance
        com_pos = self.generate_com_trajectory(target_velocity)

        return {
            'left_foot': left_foot_pos,
            'right_foot': right_foot_pos,
            'com': com_pos,
            'phase': self.current_phase,
            'support_foot': self.current_support_foot
        }

    def generate_foot_trajectory(self, foot_name, target_velocity):
        """Generate trajectory for a single foot"""
        # Determine if this foot is swing foot (moving) or stance foot (stationary)
        is_swing_foot = (foot_name != self.current_support_foot)

        if not is_swing_foot:
            # Stance foot - remains stationary or moves slowly to prepare for next step
            return self.calculate_stance_foot_position(foot_name)
        else:
            # Swing foot - follows predefined trajectory
            return self.calculate_swing_foot_trajectory(foot_name, target_velocity)

    def calculate_stance_foot_position(self, foot_name):
        """Calculate position of stance foot"""
        # Stance foot moves to prepare for next step
        # For simplicity, keep at current position with slight adjustments
        if foot_name == 'left':
            base_pos = np.array([0.0, 0.1, 0.0])  # nominal left foot position
        else:
            base_pos = np.array([0.0, -0.1, 0.0])  # nominal right foot position

        # Add small adjustments based on phase
        phase_progress = self.current_phase
        if phase_progress < self.gait_params['stance_ratio']:
            # Early stance - foot moves forward slightly
            base_pos[0] += self.gait_params['step_length'] * phase_progress / self.gait_params['stance_ratio']
        else:
            # Late stance / double support - prepare for lift-off
            base_pos[0] += self.gait_params['step_length']

        return base_pos

    def calculate_swing_foot_trajectory(self, foot_name, target_velocity):
        """Calculate trajectory for swing foot"""
        # Use cycloidal trajectory for smooth foot movement
        # Phase within the swing phase
        stance_end = self.gait_params['stance_ratio']
        double_support_end = stance_end + self.gait_params['double_support_ratio']

        if self.current_phase < stance_end or self.current_phase > double_support_end:
            # Foot should be on ground in stance phase
            return self.calculate_stance_foot_position(foot_name)
        else:
            # Calculate swing phase progress
            swing_start = stance_end
            swing_duration = 1.0 - stance_end - self.gait_params['double_support_ratio']
            swing_phase = (self.current_phase - swing_start) / swing_duration

            # Previous support foot position (where swing foot started)
            prev_support_pos = self.calculate_stance_foot_position(
                'right' if self.current_support_foot == 'left' else 'left'
            )

            # Target position (where swing foot should land)
            target_pos = prev_support_pos.copy()
            target_pos[0] += self.gait_params['step_length']  # Move forward

            # Apply target velocity adjustments
            target_pos[0] += target_velocity[0] * self.gait_params['step_duration'] / 2.0  # Forward
            target_pos[1] -= target_velocity[1] * self.gait_params['step_duration'] * 0.1  # Lateral adjustment

            # Cycloidal trajectory for smooth motion
            x_progress = self.cycloidal_interpolation(0, self.gait_params['step_length'], swing_phase)
            y_progress = self.cycloidal_interpolation(
                0, 0, swing_phase  # No lateral movement for this foot
            )
            z_progress = self.parabolic_interpolation(
                0, self.gait_params['step_height'], swing_phase
            )

            # Calculate current foot position
            foot_pos = prev_support_pos.copy()
            foot_pos[0] += x_progress
            foot_pos[1] += y_progress
            foot_pos[2] += z_progress

            return foot_pos

    def generate_com_trajectory(self, target_velocity):
        """Generate CoM trajectory for balance during walking"""
        # Generate CoM trajectory that maintains balance during walking
        # Use inverted pendulum model for CoM motion

        # Desired CoM position based on ZMP (Zero Moment Point) concept
        com_x = target_velocity[0] * self.current_phase * self.gait_params['step_duration'] / 2.0
        com_y = 0.0  # Maintain centered position
        com_z = 0.8  # Maintain constant height

        # Add small oscillations to mimic natural walking
        oscillation_freq = 1.0 / self.gait_params['step_duration']
        oscillation_amp = 0.01

        com_x += oscillation_amp * math.sin(2 * math.pi * oscillation_freq * self.current_phase * self.gait_params['step_duration'])
        com_y += oscillation_amp * 0.5 * math.sin(4 * math.pi * oscillation_freq * self.current_phase * self.gait_params['step_duration'])

        return np.array([com_x, com_y, com_z])

    def cycloidal_interpolation(self, start, end, t):
        """Cycloidal interpolation for smooth motion"""
        # Cycloidal motion: smooth acceleration/deceleration
        if t <= 0:
            return start
        elif t >= 1:
            return end
        else:
            # Cycloidal equation: s = s0 + (s1-s0) * (t - sin(2πt)/(2π))
            return start + (end - start) * (t - math.sin(2 * math.pi * t) / (2 * math.pi))

    def parabolic_interpolation(self, start, peak, t):
        """Parabolic interpolation for foot lift"""
        if t <= 0.5:
            # Rising parabola
            return start + (peak - start) * (2 * t)**2
        else:
            # Falling parabola
            return peak - (peak - start) * (2 * (t - 0.5))**2
```

## Control Integration

### Integrated Control System
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import numpy as np

class IntegratedControlSystem(Node):
    def __init__(self):
        super().__init__('integrated_control_system')

        # Initialize subsystems
        self.whole_body_controller = WholeBodyController(self.robot_model)
        self.com_controller = CoMController(self.robot_model)
        self.walking_generator = WalkingPatternGenerator(self.robot_model)

        # ROS 2 interfaces
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray, '/joint_commands', 10
        )

        # State variables
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_imu_data = None
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # [linear_x, linear_y, angular_z]

        # Control timer
        self.control_timer = self.create_timer(0.01, self.control_callback)  # 100Hz

    def joint_state_callback(self, msg):
        """Update joint state"""
        if self.current_joint_positions is None:
            self.current_joint_positions = np.zeros(len(msg.name))
            self.current_joint_velocities = np.zeros(len(msg.name))

        for i, name in enumerate(msg.name):
            try:
                idx = self.robot_model.joint_names.index(name)
                self.current_joint_positions[idx] = msg.position[i]
                if i < len(msg.velocity):
                    self.current_joint_velocities[idx] = msg.velocity[i]
            except ValueError:
                continue

    def imu_callback(self, msg):
        """Update IMU data"""
        self.current_imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def cmd_vel_callback(self, msg):
        """Update target velocity"""
        self.target_velocity = np.array([msg.linear.x, msg.linear.y, msg.angular.z])

    def control_callback(self):
        """Main control loop"""
        if (self.current_joint_positions is None or
            self.current_joint_velocities is None or
            self.current_imu_data is None):
            return

        # Update sensor data in controllers
        self.com_controller.update_sensor_data(
            self.current_imu_data,
            self.current_joint_positions,
            self.current_joint_velocities
        )

        # Generate walking pattern if walking
        walking_trajectory = None
        if np.linalg.norm(self.target_velocity[:2]) > 0.01:  # If moving
            walking_trajectory = self.walking_generator.generate_step_trajectory(
                self.target_velocity
            )

        # Define desired states for whole-body controller
        desired_states = {}

        if walking_trajectory:
            # Use walking trajectory for balance and foot placement
            desired_states['balance'] = walking_trajectory['com']
            desired_states['left_foot'] = walking_trajectory['left_foot']
            desired_states['right_foot'] = walking_trajectory['right_foot']
        else:
            # Standing balance
            desired_states['balance'] = np.array([0.0, 0.0, 0.8])  # Stable CoM position

        # Current states
        current_states = {
            'joint_positions': self.current_joint_positions,
            'joint_velocities': self.current_joint_velocities
        }

        # Compute control torques
        joint_torques = self.whole_body_controller.compute_control(
            desired_states, current_states
        )

        # Publish joint commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_torques.tolist()
        self.joint_cmd_pub.publish(cmd_msg)

        # Log control performance
        self.get_logger().debug(f'Control torques: {joint_torques[:3]}...')  # First 3 joints
```

## Exercise: Implement Control System

Create a complete control system that:
1. Integrates whole-body control with balance maintenance
2. Implements stable walking patterns
3. Handles transitions between standing and walking
4. Includes safety mechanisms for balance recovery

## Summary

The humanoid robot control systems form the foundation of stable and coordinated movement. The hierarchical control architecture ensures that high-level behaviors are properly translated into low-level joint commands while maintaining balance and safety. The integration of whole-body control, balance maintenance, and walking pattern generation enables the robot to perform complex locomotion tasks while maintaining stability.

---