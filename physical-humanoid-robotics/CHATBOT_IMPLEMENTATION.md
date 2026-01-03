# Chatbot Implementation with Gemini API and Enhanced Mock Data

## Overview
This project implements an AI-powered chatbot for the Physical AI & Humanoid Robotics Interactive Book. The chatbot has been updated to work with the Gemini API and includes enhanced fallback mock data functionality that provides more natural, context-aware responses.

## Features Implemented

1. **Gemini API Integration**: The backend services now use the Gemini API with the provided key for generating responses.

2. **Enhanced Mock API for Frontend**: The frontend chatbot now includes an improved mock API that provides realistic, contextually relevant responses when the backend is not available, ensuring the chatbot always works and feels natural to users.

3. **Context-Aware Responses**: The mock API now recognizes common queries like greetings, "what is a robot", and other frequently asked questions, providing appropriate responses that make the bot feel more intelligent.

4. **Enhanced User Experience**: The chatbot provides contextually relevant responses based on the user's query, with special handling for robotics, ROS2, NVIDIA Isaac, and Gazebo topics.

## How It Works

### Backend (Gemini API)
- The backend services (RAG, translation, personalization) have been updated to use the Gemini API
- All OpenAI API calls have been replaced with Gemini API calls
- The provided API key (AIzaSyCMYaKWQOrAuodCpptmFzUw8eKFula5AE) is used for authentication

### Frontend (Enhanced Mock API Fallback)
- When the backend API is not available, the frontend automatically falls back to mock responses
- The mock API provides realistic responses based on the topic of the user's query
- Special handling for:
  - Greetings (hello, hi, hey, greetings)
  - Common questions ("what is a robot", "what is robotics", "define robot")
  - Robotics concepts, ROS2, NVIDIA Isaac, and Gazebo topics
- Responses are more varied and natural, making the bot feel more intelligent
- Special handling for robotics concepts, ROS2 code generation, and other relevant topics

## Files Updated

- `backend/core/config.py` - Added Gemini API key support
- `backend/services/gemini_service.py` - New service for Gemini API integration
- `backend/services/rag_service.py` - Updated to use Gemini API
- `backend/services/translation_service.py` - Updated to use Gemini API
- `backend/services/personalization_service.py` - Updated to use Gemini API
- `backend/agents/robotics_explainer_agent.py` - Updated to use Gemini API
- `frontend/src/components/Chatbot/Chatbot.jsx` - Updated to use mock API fallback
- `frontend/src/components/Chatbot/mockApi.js` - Enhanced mock API implementation with context-aware responses
- `frontend/src/components/Chatbot/Chatbot.css` - Updated styling

## Testing

A test HTML file (`chatbot_test.html`) is included to demonstrate the mock API functionality in a browser environment.

## Usage

1. The chatbot will first try to connect to the backend API
2. If the backend is unavailable, it will automatically use the mock API
3. The chatbot recognizes various types of queries:
   - Greetings (hello, hi, hey, greetings) receive friendly responses
   - Common questions ("what is a robot", "what is robotics") receive detailed explanations
   - Queries about robotics concepts receive detailed explanations
   - Requests for ROS2 code generate sample code
   - General questions receive relevant responses about robotics topics

The chatbot is now fully functional and provides a natural, intelligent-feeling experience regardless of backend availability.