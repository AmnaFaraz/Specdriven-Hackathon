---
id: module-2-chapter-4
title: "Simulation to Reality Transfer (Sim-to-Real)"
sidebar_label: "Sim-to-Real Transfer"
---

# Simulation to Reality Transfer (Sim-to-Real)

Sim-to-real transfer is the process of taking behaviors, controllers, or learned policies from simulation and successfully deploying them on real robots. This is a critical challenge in robotics.

## The Reality Gap Problem

The "reality gap" refers to the differences between simulated and real environments that can cause policies learned in simulation to fail when deployed on real robots:

### Physical Differences
- **Dynamics**: Mass, friction, and inertia may not be accurately modeled
- **Actuator Behavior**: Real actuators have delays, non-linearities, and limitations
- **Sensor Noise**: Real sensors have noise, latency, and calibration errors
- **Environmental Conditions**: Lighting, temperature, and surface properties vary

### Solutions to Reality Gap
1. **Domain Randomization**: Randomize simulation parameters to improve robustness
2. **System Identification**: Calibrate simulation models to match real robot behavior
3. **Adaptive Control**: Adjust controllers based on real-world feedback
4. **Progressive Transfer**: Gradually move from simulation to reality

## Domain Randomization

Domain randomization involves training in simulations with randomized parameters:

```python
import random

class DomainRandomization:
    def __init__(self):
        self.param_ranges = {
            'robot_mass': (0.8, 1.2),  # 80% to 120% of nominal
            'friction_coeff': (0.5, 1.5),
            'sensor_noise': (0.0, 0.05),
            'actuator_delay': (0.0, 0.02),  # 0-20ms delay
        }

    def randomize_environment(self):
        """Randomize environment parameters"""
        randomized_params = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            randomized_params[param] = random.uniform(min_val, max_val)
        return randomized_params

    def apply_randomization(self, sim_env, params):
        """Apply randomized parameters to simulation"""
        sim_env.robot.mass = params['robot_mass'] * nominal_mass
        sim_env.friction_coeff = params['friction_coeff']
        sim_env.sensor_noise_level = params['sensor_noise']
        sim_env.actuator_delay = params['actuator_delay']
```

## System Identification

System identification involves calibrating simulation parameters to match real robot behavior:

```python
import numpy as np
from scipy.optimize import minimize

class SystemIdentification:
    def __init__(self, real_robot, sim_robot):
        self.real_robot = real_robot
        self.sim_robot = sim_robot
        self.parameters = {
            'mass': 1.0,
            'inertia': 0.1,
            'friction': 0.01,
            'gear_ratio': 1.0
        }

    def objective_function(self, params):
        """Objective function to minimize difference between real and sim"""
        # Set simulation parameters
        self.sim_robot.set_params({
            'mass': params[0],
            'inertia': params[1],
            'friction': params[2],
            'gear_ratio': params[3]
        })

        # Run identical commands on both robots
        real_response = self.real_robot.execute_trajectory(test_trajectory)
        sim_response = self.sim_robot.execute_trajectory(test_trajectory)

        # Calculate error
        error = np.mean((real_response - sim_response) ** 2)
        return error

    def identify_parameters(self):
        """Identify optimal parameters"""
        initial_params = [1.0, 0.1, 0.01, 1.0]  # Initial guess
        result = minimize(
            self.objective_function,
            initial_params,
            method='BFGS'
        )

        self.parameters = {
            'mass': result.x[0],
            'inertia': result.x[1],
            'friction': result.x[2],
            'gear_ratio': result.x[3]
        }
        return self.parameters
```

## Adaptive Control for Sim-to-Real

Adaptive controllers adjust their behavior based on real-world feedback:

```python
class AdaptiveController:
    def __init__(self):
        self.sim_controller = PDController()
        self.adaptation_gain = 0.01
        self.param_error_integrator = 0.0

    def update(self, state_error, sim_state, real_state):
        # Base control from simulation
        base_control = self.sim_controller.compute(state_error)

        # Adaptation based on sim-to-real discrepancy
        state_discrepancy = real_state - sim_state
        self.param_error_integrator += state_discrepancy * 0.01  # dt

        adaptation_term = self.adaptation_gain * (
            state_discrepancy + self.param_error_integrator
        )

        # Final control command
        final_control = base_control + adaptation_term

        return final_control
```

## Progressive Domain Transfer

A systematic approach to transfer from simulation to reality:

### Phase 1: Pure Simulation
- Train in idealized simulation
- Focus on basic behaviors

### Phase 2: Enhanced Simulation
- Add noise and disturbances
- Include actuator limitations
- Randomize physical parameters

### Phase 3: Reality-Matched Simulation
- Calibrate simulation to match real robot
- Use identified parameters
- Validate on real robot

### Phase 4: Reality with Safety
- Deploy on real robot with safety limits
- Monitor performance
- Collect data for improvement

## Unity-Specific Sim-to-Real Considerations

### Perception Simulation
```csharp
// In Unity, simulate sensor noise and limitations
public class SimulatedCamera : MonoBehaviour
{
    public float noiseLevel = 0.01f;
    public float blurRadius = 0.5f;

    void Update()
    {
        // Add noise to captured images
        RenderTexture.active = captureTexture;
        Texture2D image = new Texture2D(Screen.width, Screen.height);
        image.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        image.Apply();

        // Apply noise and blur to simulate real sensor limitations
        ApplyNoiseAndBlur(image);
    }

    void ApplyNoiseAndBlur(Texture2D image)
    {
        // Add random noise to pixels
        for (int x = 0; x < image.width; x++)
        {
            for (int y = 0; y < image.height; y++)
            {
                Color pixel = image.GetPixel(x, y);
                float noise = Random.Range(-noiseLevel, noiseLevel);
                pixel.r += noise;
                pixel.g += noise;
                pixel.b += noise;
                image.SetPixel(x, y, pixel);
            }
        }
    }
}
```

## Exercise: Implement Domain Randomization

Create a simulation environment with randomized parameters for:
1. Robot mass (±20%)
2. Joint friction (±50%)
3. Sensor noise levels
4. Actuator delays

Train a simple controller in this randomized environment and test its performance on a fixed parameter environment.

## Summary

Sim-to-real transfer is essential for making simulation useful in real robotics applications. By using domain randomization, system identification, adaptive control, and progressive transfer techniques, we can bridge the reality gap and successfully deploy simulation-trained behaviors on real robots.

---