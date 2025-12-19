import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PersonalizationToggle = ({ chapterId }) => {
  const [difficulty, setDifficulty] = useState('intermediate');
  const [contentFocus, setContentFocus] = useState('theoretical');
  const [isPersonalized, setIsPersonalized] = useState(false);
  const [loading, setLoading] = useState(false);

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

  useEffect(() => {
    // Load saved preferences for this chapter
    loadPreferences();
  }, [chapterId]);

  const loadPreferences = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/personalization/${chapterId}`, {
        params: { user_id: localStorage.getItem('user_id') || '1' }
      });

      if (response.data) {
        setDifficulty(response.data.difficulty_level);
        setContentFocus(response.data.content_focus);
        setIsPersonalized(true);
      }
    } catch (error) {
      console.log('No saved preferences for this chapter');
    }
  };

  const handleSavePreferences = async () => {
    setLoading(true);
    try {
      await axios.put(`${API_BASE_URL}/api/personalization/${chapterId}`, {
        user_id: localStorage.getItem('user_id') || '1',
        difficulty_level: difficulty,
        content_focus: contentFocus,
        language_preference: 'en'
      });

      setIsPersonalized(true);
      alert('Personalization settings saved!');
    } catch (error) {
      console.error('Error saving preferences:', error);
      alert('Error saving preferences');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="personalization-controls">
      <h4>Personalize Content</h4>

      <div className="personalization-option">
        <label>Difficulty Level:</label>
        <select
          value={difficulty}
          onChange={(e) => setDifficulty(e.target.value)}
          disabled={loading}
        >
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="advanced">Advanced</option>
        </select>
      </div>

      <div className="personalization-option">
        <label>Content Focus:</label>
        <select
          value={contentFocus}
          onChange={(e) => setContentFocus(e.target.value)}
          disabled={loading}
        >
          <option value="theoretical">Theoretical</option>
          <option value="practical">Practical</option>
          <option value="application">Application</option>
        </select>
      </div>

      <button
        className="personalization-toggle"
        onClick={handleSavePreferences}
        disabled={loading}
      >
        {loading ? 'Saving...' : isPersonalized ? 'Update Settings' : 'Apply Personalization'}
      </button>

      {isPersonalized && (
        <div className="personalization-status">
          ✓ Applied: {difficulty} level, {contentFocus} focus
        </div>
      )}
    </div>
  );
};

export default PersonalizationToggle;