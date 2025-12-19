import React, { useState } from 'react';
import axios from 'axios';

const TranslationToggle = ({ chapterId, content, onContentChange }) => {
  const [isTranslating, setIsTranslating] = useState(false);
  const [isTranslated, setIsTranslated] = useState(false);
  const [originalContent, setOriginalContent] = useState(content);

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

  const toggleTranslation = async () => {
    if (isTranslated) {
      // Switch back to original content
      onContentChange(originalContent);
      setIsTranslated(false);
    } else {
      // Translate to Urdu
      setIsTranslating(true);
      try {
        const response = await axios.post(`${API_BASE_URL}/api/translate`, {
          text: content,
          target_language: 'ur',
          source_language: 'en',
          context: `Chapter ${chapterId}`,
          user_id: localStorage.getItem('user_id') || null
        });

        onContentChange(response.data.translated_text);
        setIsTranslated(true);
      } catch (error) {
        console.error('Translation error:', error);
        alert('Error translating content. Please try again.');
      } finally {
        setIsTranslating(false);
      }
    }
  };

  return (
    <button
      className="translation-toggle"
      onClick={toggleTranslation}
      disabled={isTranslating}
    >
      {isTranslating
        ? 'Translating...'
        : isTranslated
          ? 'Show in English'
          : 'Translate to Urdu'}
    </button>
  );
};

export default TranslationToggle;