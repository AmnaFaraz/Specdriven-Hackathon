import React, { useState } from 'react';
import axios from 'axios';

const SubagentInterface = ({ chapterId, content }) => {
  const [selectedAgent, setSelectedAgent] = useState('');
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

  const availableAgents = [
    { value: 'robotics_explainer', label: 'Robotics Explainer', description: 'Get detailed explanations of robotics concepts' },
    { value: 'ros2_code', label: 'ROS2 Code Generator', description: 'Generate ROS2 Python code for your tasks' },
    { value: 'urdu_translator', label: 'Urdu Translator', description: 'Translate content to Urdu' },
    { value: 'personalization', label: 'Content Personalizer', description: 'Adapt content to your learning preferences' }
  ];

  const executeSubagent = async () => {
    if (!selectedAgent || !query) {
      setError('Please select an agent and enter a query');
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/subagent/execute`, {
        query,
        agent_type: selectedAgent,
        user_id: localStorage.getItem('user_id') || null,
        context: `Chapter ${chapterId}`,
        user_preferences: {
          background: localStorage.getItem('background') || '',
          difficulty_level: localStorage.getItem('difficulty_level') || 'intermediate',
          content_focus: localStorage.getItem('content_focus') || 'theoretical'
        }
      });

      setResult(response.data);
    } catch (error) {
      console.error('Subagent execution error:', error);
      setError('Error executing subagent. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="subagent-interface">
      <h4>Specialized AI Assistance</h4>

      <div className="subagent-controls">
        <label>Select Agent:</label>
        <select
          value={selectedAgent}
          onChange={(e) => setSelectedAgent(e.target.value)}
          disabled={isLoading}
        >
          <option value="">Choose an agent...</option>
          {availableAgents.map(agent => (
            <option key={agent.value} value={agent.value}>
              {agent.label}
            </option>
          ))}
        </select>

        {selectedAgent && (
          <div className="agent-description">
            {availableAgents.find(a => a.value === selectedAgent)?.description}
          </div>
        )}

        <label>Query:</label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query here..."
          rows="4"
          disabled={isLoading}
          style={{ width: '100%', marginBottom: '10px' }}
        />

        <button onClick={executeSubagent} disabled={isLoading || !selectedAgent || !query}>
          {isLoading ? 'Processing...' : 'Execute Agent'}
        </button>

        {error && <div className="error-message">{error}</div>}
      </div>

      {result && (
        <div className="subagent-result">
          <h5>Result from {result.agent_type}:</h5>
          <div className="result-content">
            {result.result.explanation && (
              <div>
                <h6>Explanation:</h6>
                <p>{result.result.explanation}</p>
              </div>
            )}
            {result.result.generated_code && (
              <div>
                <h6>Generated Code:</h6>
                <pre className="code-block">
                  {result.result.generated_code}
                </pre>
              </div>
            )}
            {result.result.translated_text && (
              <div>
                <h6>Translation:</h6>
                <p>{result.result.translated_text}</p>
              </div>
            )}
            {result.result.personalized_content && (
              <div>
                <h6>Personalized Content:</h6>
                <p>{result.result.personalized_content}</p>
              </div>
            )}
            {result.result.query && !result.result.explanation && (
              <p>{result.result.query}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default SubagentInterface;