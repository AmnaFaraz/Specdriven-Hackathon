---
id: module-2-chapter-3
title: "Unity: High-Fidelity Visualization and Simulation"
sidebar_label: "Unity Simulation"
---

# Unity: High-Fidelity Visualization and Simulation

Unity provides high-fidelity visualization and simulation capabilities that complement physics-based simulators like Gazebo, offering photorealistic rendering and advanced graphics features.

## Unity for Robotics Overview

Unity Robotics provides:
- **Photorealistic Rendering**: High-quality visuals for realistic perception simulation
- **XR Support**: Virtual and augmented reality capabilities
- **Asset Store**: Extensive library of 3D models and environments
- **C# Scripting**: Alternative to C++/Python for robotics applications
- **Cross-Platform**: Deploy to multiple platforms including mobile and VR

## Unity Robotics Hub

The Unity Robotics Hub provides essential tools:
- **ROS-TCP-Connector**: Bridge between Unity and ROS/ROS2
- **Unity Perception**: Tools for synthetic data generation
- **ML-Agents**: Reinforcement learning framework
- **Visual Scripting**: No-code development tools

## Setting up Unity for Robotics

### Installation
1. Download Unity Hub from unity.com
2. Install Unity 2022.3 LTS or newer
3. Install required packages via Unity Package Manager

### ROS-TCP-Connector Setup
```csharp
// Example Unity C# script for ROS communication
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string rosIP = "127.0.0.1";
    int rosPort = 10000;

    // Robot joint angles
    float[] jointAngles = new float[18];

    void Start()
    {
        // Connect to ROS
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);

        // Start publishing joint states
        InvokeRepeating("PublishJointStates", 0.0f, 0.05f); // 20 Hz
    }

    void PublishJointStates()
    {
        // Update joint angles (example with simple oscillation)
        for (int i = 0; i < jointAngles.Length; i++)
        {
            jointAngles[i] = Mathf.Sin(Time.time + i * 0.5f) * 0.5f;
        }

        // Create and publish joint state message
        var jointState = new JointStateMsg();
        jointState.name = new string[] {
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
            "left_knee", "left_ankle_pitch", "left_ankle_roll",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
            "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
            "right_shoulder_pitch", "right_shoulder_roll", "right_elbow"
        };
        jointState.position = jointAngles;
        jointState.velocity = new float[jointAngles.Length];
        jointState.effort = new float[jointAngles.Length];

        ros.Publish("/joint_states", jointState);
    }

    void OnJointStateMessage(JointStateMsg jointState)
    {
        // Process incoming joint state messages
        for (int i = 0; i < jointState.name.Length; i++)
        {
            // Update robot model based on joint positions
            UpdateJoint(jointState.name[i], jointState.position[i]);
        }
    }

    void UpdateJoint(string jointName, float angle)
    {
        // Find the joint in the hierarchy and rotate it
        Transform joint = transform.Find(jointName);
        if (joint != null)
        {
            // Apply rotation based on joint type
            joint.localRotation = Quaternion.Euler(0, angle * Mathf.Rad2Deg, 0);
        }
    }
}
```

## Unity Perception Package

Unity Perception enables synthetic data generation for AI training:

```csharp
using Unity.Perception.GroundTruth;
using Unity.Perception.Labeling;

public class PerceptionSetup : MonoBehaviour
{
    void Start()
    {
        // Register the dataset capture component
        var datasetCapture = gameObject.AddComponent<DatasetCapture>();

        // Configure capture settings
        datasetCapture.CaptureSegmentationLabels = true;
        datasetCapture.CaptureDepth = true;
        datasetCapture.CaptureOcclusion = true;

        // Set up semantic segmentation
        var semanticSegmentation = gameObject.AddComponent<SemanticSegmentationLabeler>();
    }
}
```

## ML-Agents for Robot Learning

Unity ML-Agents enables reinforcement learning for humanoid robots:

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class HumanoidAgent : Agent
{
    [SerializeField] Transform target;
    [SerializeField] float moveSpeed = 5f;
    [SerializeField] float rotationSpeed = 100f;

    Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // Reset agent position
        transform.position = new Vector3(Random.Range(-5f, 5f), 1f, Random.Range(-5f, 5f));
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Agent position and rotation
        sensor.AddObservation(transform.position);
        sensor.AddObservation(transform.rotation);

        // Target position
        sensor.AddObservation(target.position);

        // Velocity
        sensor.AddObservation(rb.velocity);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Extract actions
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        float rotation = actions.ContinuousActions[2];

        // Apply movement
        Vector3 moveDirection = new Vector3(moveX, 0, moveZ).normalized;
        rb.AddForce(moveDirection * moveSpeed, ForceMode.Acceleration);
        transform.Rotate(Vector3.up, rotation * rotationSpeed * Time.deltaTime);

        // Reward system
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        SetReward(-distanceToTarget * 0.01f); // Negative reward for distance

        // End episode if close to target
        if (distanceToTarget < 1.5f)
        {
            SetReward(1f);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
        continuousActionsOut[2] = Input.GetAxis("Rotate");
    }
}
```

## Exercise: Create a Simple Unity Robot Controller

Create a Unity scene with:
1. A simple humanoid robot model
2. Joint control system
3. Basic ROS communication
4. Perception components for data collection

## Summary

Unity provides high-fidelity visualization and simulation capabilities that complement physics-based simulators. Its photorealistic rendering, XR support, and machine learning frameworks make it valuable for perception system development and AI training in robotics applications.

---