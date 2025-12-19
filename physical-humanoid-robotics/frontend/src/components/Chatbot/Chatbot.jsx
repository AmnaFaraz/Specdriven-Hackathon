import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Chatbot.css';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = { text: inputValue, sender: 'user', timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Check if this is a subagent command
      const lowerInput = inputValue.toLowerCase();
      if (lowerInput.includes('robotics') && (lowerInput.includes('explain') || lowerInput.includes('concept'))) {
        // Use robotics explainer agent
        const response = await axios.post(`${API_BASE_URL}/api/subagent/execute`, {
          query: inputValue,
          agent_type: 'robotics_explainer',
          user_id: localStorage.getItem('user_id') || null,
          user_preferences: {
            background: localStorage.getItem('background') || ''
          }
        });

        const botMessage = {
          text: response.data.result.explanation || response.data.result.query,
          sender: 'bot',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else if (lowerInput.includes('ros2') && (lowerInput.includes('code') || lowerInput.includes('generate'))) {
        // Use ROS2 code agent
        const response = await axios.post(`${API_BASE_URL}/api/subagent/execute`, {
          query: inputValue,
          agent_type: 'ros2_code',
          user_id: localStorage.getItem('user_id') || null
        });

        const botMessage = {
          text: `Here's the generated ROS2 code:\n\n\`\`\`python\n${response.data.result.generated_code}\n\`\`\``,
          sender: 'bot',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        // Use regular RAG query
        // Get selected text if any
        const selectedText = window.getSelection().toString();

        const response = await axios.post(`${API_BASE_URL}/api/rag/query`, {
          query: inputValue,
          context: selectedText ? 'selected_text' : 'entire_book',
          selected_text: selectedText || null,
          user_id: localStorage.getItem('user_id') || null
        });

        const botMessage = {
          text: response.data.response,
          sender: 'bot',
          sources: response.data.sources,
          timestamp: new Date()
        };

        setMessages(prev => [...prev, botMessage]);
      }
    } catch (error) {
      const errorMessage = {
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chatbot-container">
      {isOpen ? (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <h3>AI Assistant</h3>
            <button className="chatbot-close" onClick={() => setIsOpen(false)}>
              ×
            </button>
          </div>
          <div className="chatbot-messages">
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.sender}`}>
                <div className="message-text">{message.text}</div>
                {message.sources && message.sources.length > 0 && (
                  <div className="message-sources">
                    <strong>Sources:</strong>
                    <ul>
                      {message.sources.map((source, idx) => (
                        <li key={idx}>
                          {source.title || `Chapter ${source.chapter_id}`}: {source.content.substring(0, 100)}...
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
            {isLoading && <div className="message bot">Thinking...</div>}
          </div>
          <div className="chatbot-input">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about the book content..."
              rows="3"
            />
            <button onClick={handleSendMessage} disabled={isLoading}>
              Send
            </button>
          </div>
        </div>
      ) : (
        <button className="chatbot-toggle" onClick={() => setIsOpen(true)}>
          💬 AI Assistant
        </button>
      )}
    </div>
  );
};

export default Chatbot;