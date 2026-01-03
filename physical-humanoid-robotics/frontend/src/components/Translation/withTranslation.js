import React, { useState, useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

// A higher-order component that wraps content with translation functionality
export default function withTranslation(WrappedComponent) {
  return function TranslatableComponent(props) {
    const [translatedContent, setTranslatedContent] = useState(null);
    const [isTranslated, setIsTranslated] = useState(false);
    const [isTranslating, setIsTranslating] = useState(false);

    // Extract content for translation
    const extractContent = () => {
      // This is a simplified approach - in practice, you'd extract the actual content
      return props.content?.metadata?.title || '';
    };

    const translateContent = async () => {
      setIsTranslating(true);
      try {
        // This would call the actual translation API
        const response = await fetch('/api/translate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: extractContent(),
            target_language: 'ur',
            source_language: 'en'
          })
        });

        if (response.ok) {
          const data = await response.json();
          setTranslatedContent(data.translated_text);
          setIsTranslated(true);
        }
      } catch (error) {
        console.error('Translation error:', error);
      } finally {
        setIsTranslating(false);
      }
    };

    const toggleTranslation = () => {
      if (isTranslated) {
        setIsTranslated(false);
        setTranslatedContent(null);
      } else {
        translateContent();
      }
    };

    return (
      <div>
        <div style={{
          position: 'sticky',
          top: '20px',
          right: '20px',
          zIndex: '999',
          display: 'flex',
          gap: '10px',
          justifyContent: 'flex-end'
        }}>
          <button
            onClick={toggleTranslation}
            disabled={isTranslating}
            style={{
              padding: '8px 16px',
              backgroundColor: '#4a00e0',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {isTranslating ? 'Translating...' : isTranslated ? 'Show English' : 'Translate to Urdu'}
          </button>
        </div>

        {isTranslated && translatedContent ? (
          <div className="translated-content">
            {translatedContent}
          </div>
        ) : (
          <WrappedComponent {...props} />
        )}
      </div>
    );
  };
}