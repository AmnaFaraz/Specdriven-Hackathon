/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'introduction',
    'conclusion',
    {
      type: 'category',
      label: 'Modules',
      items: [
        {
          type: 'category',
          label: 'Module 1: The Robotic Nervous System (ROS 2)',
          items: [
            'modules/module-1/module-1-chapter-1',
            'modules/module-1/module-1-chapter-2',
            'modules/module-1/module-1-chapter-3',
            'modules/module-1/module-1-chapter-4',
            'modules/module-1/module-1-chapter-5',
          ],
        },
        {
          type: 'category',
          label: 'Module 2: Digital Twin (Gazebo & Unity)',
          items: [
            'modules/module-2/module-2-chapter-1',
            'modules/module-2/module-2-chapter-2',
            'modules/module-2/module-2-chapter-3',
            'modules/module-2/module-2-chapter-4',
            'modules/module-2/module-2-chapter-5',
          ],
        },
        {
          type: 'category',
          label: 'Module 3: AI-Brain (NVIDIA Isaac)',
          items: [
            'modules/module-3/module-3-chapter-1',
            'modules/module-3/module-3-chapter-2',
            'modules/module-3/module-3-chapter-3',
            'modules/module-3/module-3-chapter-4',
            'modules/module-3/module-3-chapter-5',
          ],
        },
        {
          type: 'category',
          label: 'Module 4: Vision-Language-Action (VLA)',
          items: [
            'modules/module-4/module-4-chapter-1',
            'modules/module-4/module-4-chapter-2',
            'modules/module-4/module-4-chapter-3',
            'modules/module-4/module-4-chapter-4',
            'modules/module-4/module-4-chapter-5',
          ],
        },
        {
          type: 'category',
          label: 'Module 5: Capstone - Autonomous Humanoid Project',
          items: [
            'modules/module-5/module-5-chapter-1',
            'modules/module-5/module-5-chapter-2',
            'modules/module-5/module-5-chapter-3',
            'modules/module-5/module-5-chapter-4',
            'modules/module-5/module-5-chapter-5',
          ],
        },
      ],
    },
  ],
  moduleSidebar: [
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        {
          type: 'category',
          label: 'Chapter 1: Introduction to ROS 2 for Humanoids',
          items: [
            'modules/module-1/module-1-chapter-1',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: ROS 2 Communication Patterns',
          items: [
            'modules/module-1/module-1-chapter-2',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 3: ROS 2 Navigation and Control',
          items: [
            'modules/module-1/module-1-chapter-3',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 4: ROS 2 Perception and Sensing',
          items: [
            'modules/module-1/module-1-chapter-4',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 5: ROS 2 Integration and Testing',
          items: [
            'modules/module-1/module-1-chapter-5',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin (Gazebo & Unity)',
      items: [
        {
          type: 'category',
          label: 'Chapter 1: Digital Twin Fundamentals',
          items: [
            'modules/module-2/module-2-chapter-1',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: Gazebo Simulation for Humanoids',
          items: [
            'modules/module-2/module-2-chapter-2',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 3: Unity Integration and VR',
          items: [
            'modules/module-2/module-2-chapter-3',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 4: Simulation to Reality Transfer',
          items: [
            'modules/module-2/module-2-chapter-4',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 5: Digital Twin Validation',
          items: [
            'modules/module-2/module-2-chapter-5',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI-Brain (NVIDIA Isaac)',
      items: [
        {
          type: 'category',
          label: 'Chapter 1: Introduction to NVIDIA Isaac for Robotics',
          items: [
            'modules/module-3/module-3-chapter-1',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: Isaac Perception and Understanding',
          items: [
            'modules/module-3/module-3-chapter-2',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 3: Isaac AI Planning and Navigation',
          items: [
            'modules/module-3/module-3-chapter-3',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 4: Isaac Manipulation and Control',
          items: [
            'modules/module-3/module-3-chapter-4',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 5: Isaac Learning and Adaptation',
          items: [
            'modules/module-3/module-3-chapter-5',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        {
          type: 'category',
          label: 'Chapter 1: Introduction to Vision-Language-Action Systems',
          items: [
            'modules/module-4/module-4-chapter-1',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: Implementing Vision-Language Models for Robotics',
          items: [
            'modules/module-4/module-4-chapter-2',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 3: AI Perception and Decision Making',
          items: [
            'modules/module-4/module-4-chapter-3',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 4: Humanoid Robot Integration and Testing',
          items: [
            'modules/module-4/module-4-chapter-4',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 5: Deploying VLA Systems in Real-World Robotics',
          items: [
            'modules/module-4/module-4-chapter-5',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 5: Capstone - Autonomous Humanoid Project',
      items: [
        {
          type: 'category',
          label: 'Chapter 1: Autonomous Humanoid Project Overview',
          items: [
            'modules/module-5/module-5-chapter-1',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: Humanoid Robot Control Systems',
          items: [
            'modules/module-5/module-5-chapter-2',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 3: [To be created]',
          items: [
            'modules/module-5/module-5-chapter-3',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 4: [To be created]',
          items: [
            'modules/module-5/module-5-chapter-4',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 5: Complete Humanoid Robot System Integration',
          items: [
            'modules/module-5/module-5-chapter-5',
          ],
        },
      ],
    },
  ],
};

module.exports = sidebars;