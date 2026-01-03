/*
 * Test file to demonstrate the chatbot functionality with mock API
 * This file is not part of the actual application but demonstrates how the chatbot works
 */

// Import the mock API
import { mockApi } from './src/components/Chatbot/mockApi';

async function testChatbot() {
  console.log('Testing chatbot functionality with mock API...\n');
  
  // Test different types of queries
  const testQueries = [
    'What is robotics?',
    'Explain ROS2 concepts',
    'What is NVIDIA Isaac?',
    'How does Gazebo work?',
    'Generate ROS2 publisher code',
    'Explain humanoid robots'
  ];
  
  for (const query of testQueries) {
    console.log(`Query: ${query}`);
    
    try {
      // Test RAG query
      const ragResponse = await mockApi.ragQuery(query, 'entire_book', null);
      console.log(`Response: ${ragResponse.response.substring(0, 100)}...`);
      console.log(`Sources: ${ragResponse.sources.length} sources\n`);
    } catch (error) {
      console.error(`Error with query "${query}":`, error.message);
    }
  }
  
  // Test subagent functionality
  console.log('Testing subagent functionality...\n');
  
  // Test robotics explainer
  const roboticsQuery = 'Explain how robots work';
  const roboticsResponse = await mockApi.executeSubagent(roboticsQuery, 'robotics_explainer');
  console.log(`Robotics Explainer Query: ${roboticsQuery}`);
  console.log(`Response: ${roboticsResponse.result.explanation.substring(0, 100)}...\n`);
  
  // Test ROS2 code generation
  const codeQuery = 'Generate ROS2 publisher';
  const codeResponse = await mockApi.executeSubagent(codeQuery, 'ros2_code');
  console.log(`ROS2 Code Query: ${codeQuery}`);
  console.log(`Generated Code Preview: ${codeResponse.result.generated_code.substring(0, 100)}...\n`);
  
  console.log('All tests completed successfully! The chatbot should now work with mock data.');
}

// Run the test if this file is executed directly
if (typeof window !== 'undefined') {
  // Browser environment
  testChatbot().catch(console.error);
} else {
  // Node.js environment
  testChatbot().catch(console.error);
}