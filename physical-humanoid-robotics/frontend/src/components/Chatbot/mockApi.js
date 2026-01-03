// Mock API responses for the chatbot
const mockResponses = {
  // Greeting responses
  greeting: [
    "Hello! I'm your AI assistant for the Physical AI & Humanoid Robotics Interactive Book. How can I help you today?",
    "Hi there! I'm here to answer any questions you have about robotics, AI, and humanoid systems. What would you like to know?",
    "Greetings! I'm your robotics knowledge assistant. Feel free to ask me anything about robotics, ROS2, NVIDIA Isaac, or related topics.",
    "Hello! I'm excited to help you explore the fascinating world of robotics. What can I assist you with?",
    "Hi! I'm your AI guide for the Physical AI & Humanoid Robotics book. Ask me anything about the content!"
  ],

  // General responses
  general: [
    "The field of robotics combines engineering and computer science to design, construct, and operate robots. These machines can assist humans in various tasks, from manufacturing to space exploration.",
    "Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, computer science, and others. It deals with the design, construction, operation, and use of robots, as well as computer systems for their control, sensory feedback, and information processing.",
    "Humanoid robots are robots with physical features resembling human bodies. They typically have a head, torso, two arms, and two legs, and may have a face with eyes and mouth.",
    "NVIDIA Isaac is a robotics platform that provides tools and technologies for developing, simulating, and deploying AI-powered robots. It includes Isaac Sim for simulation and Isaac ROS for perception and navigation.",
    "ROS 2 (Robot Operating System 2) is flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.",
    "Gazebo is a robot simulator that provides realistic 3D simulation environments for robotics development. It allows testing of algorithms, training of robots, and experimentation without the need for physical hardware.",
    "Artificial intelligence in robotics enables machines to perceive their environment, make decisions, and perform tasks with varying degrees of autonomy. This includes perception, planning, control, and learning capabilities.",
    "The integration of AI and robotics is transforming industries by enabling machines to perform complex tasks that previously required human intelligence and dexterity.",
    "Modern robotics combines mechanical engineering, electronics, and software to create machines that can interact with their environment and perform useful tasks.",
    "Humanoid robots represent one of the most ambitious goals in robotics, attempting to create machines that can interact with humans in a natural, intuitive way.",
    "The field of robotics is rapidly evolving with advances in AI, sensors, actuators, and materials science. Today's robots are more capable and accessible than ever before.",
    "Robots are increasingly being used in healthcare for surgery, rehabilitation, and assistance for elderly or disabled individuals.",
    "The future of robotics includes more autonomous systems, human-robot collaboration, and robots that can adapt to new situations without explicit programming.",
    "Safety is a critical consideration in robotics, especially as robots become more integrated into human environments and workplaces.",
    "Robotics research focuses on challenges like perception, manipulation, navigation, human-robot interaction, and autonomous decision-making."
  ],

  // Responses for robotics-related questions
  robotics: [
    "Robotics is an interdisciplinary field that encompasses mechanical engineering, electrical engineering, and computer science. It involves the design, construction, operation, and use of robots.",
    "A robot is a programmable machine that can execute tasks autonomously or semi-autonomously. Modern robots often incorporate artificial intelligence to make decisions based on sensor input.",
    "Robotics applications span across various industries including manufacturing, healthcare, agriculture, space exploration, and domestic services.",
    "The main components of a robot include sensors (to perceive the environment), actuators (to move), a control system (to process information), and a power source.",
    "A robot is a programmable machine that can execute tasks automatically. It typically includes sensors to perceive its environment, actuators to move, and a control system to process information and make decisions.",
    "Robots can be classified in various ways: by application (industrial, service, medical), by mobility (mobile, stationary), by control method (autonomous, teleoperated), or by physical form (humanoid, wheeled, flying).",
    "The history of robotics dates back to ancient times with automata, but modern robotics began in the 1950s with industrial robots. Today, robots are becoming increasingly sophisticated with AI integration.",
    "Robotics is an interdisciplinary field combining mechanical engineering, electrical engineering, and computer science to design, construct, operate, and apply robots.",
    "The three laws of robotics, coined by Isaac Asimov, are: 1) A robot may not injure a human or allow a human to come to harm, 2) A robot must obey human orders unless it conflicts with law 1, 3) A robot must protect its own existence unless it conflicts with laws 1 or 2.",
    "A robot is a programmable machine that can execute tasks automatically. It typically includes sensors to perceive its environment, actuators to move, and a control system to process information and make decisions.",
    "Modern robots use various types of sensors including cameras, LIDAR, ultrasonic sensors, and tactile sensors to perceive their environment.",
    "Robots are used in manufacturing for tasks like assembly, painting, welding, and material handling, improving efficiency and safety.",
    "Service robots assist humans in non-industrial tasks, including cleaning, delivery, healthcare assistance, and entertainment.",
    "Mobile robots can move around in their environment using wheels, tracks, legs, or other locomotion methods to perform tasks in different locations.",
    "Humanoid robots are designed to resemble and mimic human behavior, often featuring a head, torso, arms, and legs to interact naturally with human environments."
  ],

  // Responses for ROS2-related questions
  ros2: [
    "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.",
    "ROS 2 provides several key features including message passing between nodes, package management, and tools for debugging and visualization.",
    "In ROS 2, nodes are processes that perform computation. Nodes are organized into packages for sharing and reuse.",
    "ROS 2 uses a DDS (Data Distribution Service) implementation for communication between nodes, which provides a publish-subscribe communication pattern.",
    "ROS 2 is designed for production environments with improved security, real-time capabilities, and support for commercial development.",
    "ROS 2 introduces quality of service (QoS) policies that allow fine-tuning of communication between nodes based on requirements like reliability and latency.",
    "The ROS 2 ecosystem includes tools for simulation (Gazebo), visualization (RViz), and debugging (rqt), making it a comprehensive development platform.",
    "ROS 2 supports multiple DDS implementations like Fast DDS, Cyclone DDS, and RTI Connext DDS, providing flexibility in communication middleware.",
    "ROS 2 packages are organized using the colcon build system, which allows for efficient building of large-scale robotic applications."
  ],

  // Responses for NVIDIA Isaac-related questions
  isaac: [
    "NVIDIA Isaac is a robotics platform that provides tools and technologies for developing, simulating, and deploying AI-powered robots.",
    "Isaac Sim is NVIDIA's robotics simulator that provides realistic 3D simulation environments for robotics development.",
    "Isaac ROS is a collection of packages that accelerate perception and navigation workloads for robotics applications.",
    "Isaac provides tools for training and deploying AI models for robotics applications, leveraging NVIDIA's GPU computing capabilities.",
    "The NVIDIA Isaac platform includes Isaac Sim for simulation, Isaac ROS for perception and navigation, Isaac Apps for reference applications, and Isaac Lab for reinforcement learning.",
    "Isaac Sim provides photorealistic simulation with accurate physics, enabling developers to train and test robots in virtual environments before deploying on real hardware.",
    "Isaac ROS includes hardware acceleration for perception tasks like SLAM, object detection, and depth estimation using NVIDIA GPUs.",
    "NVIDIA Isaac enables the development of AI-powered robots with capabilities like perception, navigation, manipulation, and learning."
  ],

  // Responses for Gazebo-related questions
  gazebo: [
    "Gazebo is a robot simulator that provides realistic 3D simulation environments for robotics development.",
    "Gazebo provides physics simulation, sensor simulation, and realistic rendering capabilities for testing robots in virtual environments.",
    "Gazebo allows developers to test robot algorithms without the need for physical hardware, reducing development time and costs.",
    "Gazebo integrates with ROS/ROS2 to provide seamless simulation and real-world deployment capabilities.",
    "Gazebo simulates realistic physics, sensors, and environments to enable testing of robot algorithms before deployment on real hardware.",
    "Gazebo supports various physics engines including ODE, Bullet, Simbody, and DART for accurate simulation of robot dynamics.",
    "Gazebo includes a model database with thousands of pre-built robot models and environments for simulation.",
    "Gazebo enables testing of robot behaviors in complex scenarios that would be difficult or dangerous to test with real robots."
  ],

  // Responses for VLA (Vision-Language-Action) questions
  vla: [
    "Vision-Language-Action (VLA) models combine visual perception, language understanding, and action execution in robotics.",
    "VLA models enable robots to understand natural language commands and perform corresponding physical actions in real-world environments.",
    "These models integrate computer vision, natural language processing, and robotic control to create more intuitive human-robot interaction.",
    "VLA models represent a significant advancement in making robots more accessible to non-expert users through natural language commands.",
    "VLA models allow robots to follow complex instructions by combining visual understanding with language processing to execute appropriate actions.",
    "Vision-Language-Action models are trained on large datasets of human demonstrations to learn the connection between language, perception, and action."
  ]
};

// Mock ROS2 code examples
const mockRos2Code = {
  publisher: `import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()`,
  
  subscriber: `import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()`
};

// Function to get a random response based on the topic
function getMockResponse(query) {
  const lowerQuery = query.toLowerCase().trim();

  // Handle greetings
  if (lowerQuery.includes('hello') || lowerQuery.includes('hi') || lowerQuery.includes('hey') || lowerQuery.includes('greetings')) {
    return mockResponses.greeting[Math.floor(Math.random() * mockResponses.greeting.length)];
  }
  // Handle "what is" questions
  else if (lowerQuery.includes('what is a robot') || lowerQuery.includes('what is robotics') || lowerQuery.includes('define robot') || lowerQuery.includes('explain robot')) {
    return mockResponses.robotics[Math.floor(Math.random() * mockResponses.robotics.length)];
  }
  // Handle ROS2 related queries
  else if (lowerQuery.includes('ros') || lowerQuery.includes('ros2')) {
    return mockResponses.ros2[Math.floor(Math.random() * mockResponses.ros2.length)];
  }
  // Handle NVIDIA Isaac related queries
  else if (lowerQuery.includes('isaac') || lowerQuery.includes('nvidia')) {
    return mockResponses.isaac[Math.floor(Math.random() * mockResponses.isaac.length)];
  }
  // Handle Gazebo related queries
  else if (lowerQuery.includes('gazebo')) {
    return mockResponses.gazebo[Math.floor(Math.random() * mockResponses.gazebo.length)];
  }
  // Handle VLA related queries
  else if (lowerQuery.includes('vla') || lowerQuery.includes('vision-language') || lowerQuery.includes('vision language')) {
    return mockResponses.vla[Math.floor(Math.random() * mockResponses.vla.length)];
  }
  // Handle robotics related queries
  else if (lowerQuery.includes('robot') || lowerQuery.includes('robotics')) {
    return mockResponses.robotics[Math.floor(Math.random() * mockResponses.robotics.length)];
  }
  // Default to general response
  else {
    return mockResponses.general[Math.floor(Math.random() * mockResponses.general.length)];
  }
}

// Function to get mock ROS2 code
function getMockRos2Code(query) {
  if (query.toLowerCase().includes('publisher')) {
    return mockRos2Code.publisher;
  } else if (query.toLowerCase().includes('subscriber')) {
    return mockRos2Code.subscriber;
  } else {
    // Return a random code example
    const keys = Object.keys(mockRos2Code);
    const randomKey = keys[Math.floor(Math.random() * keys.length)];
    return mockRos2Code[randomKey];
  }
}

// Mock API functions
export const mockApi = {
  // Mock RAG query
  async ragQuery(query, context, selectedText) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    return {
      response: getMockResponse(query),
      sources: [
        {
          chapter_id: "1.1",
          title: "Introduction to Robotics",
          content: "Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, computer science, and others.",
          similarity_score: 0.95
        },
        {
          chapter_id: "2.3",
          title: "ROS 2 Fundamentals",
          content: "ROS 2 (Robot Operating System 2) is flexible framework for writing robot software. It's a collection of tools, libraries, and conventions.",
          similarity_score: 0.87
        }
      ]
    };
  },
  
  // Mock subagent execution
  async executeSubagent(query, agentType) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    if (agentType === 'robotics_explainer') {
      return {
        result: {
          explanation: getMockResponse(query),
          concept: query,
          background_applied: true
        }
      };
    } else if (agentType === 'ros2_code') {
      return {
        result: {
          generated_code: getMockRos2Code(query),
          query: query
        }
      };
    } else {
      return {
        result: {
          response: getMockResponse(query),
          query: query
        }
      };
    }
  }
};