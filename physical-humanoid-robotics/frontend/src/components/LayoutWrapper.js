import React from 'react';
import Chatbot from './Chatbot/Chatbot';
import TranslationToggle from './Translation/TranslationToggle';

// A layout wrapper that adds chatbot and translation to the page
const LayoutWrapper = ({ children, metadata = {} }) => {
  // Extract content for translation
  const contentForTranslation = metadata?.title ? `${metadata.title}\n\n${metadata.description || ''}` : '';

  return (
    <div style={{ position: 'relative' }}>
      {/* Translation Toggle - Top right */}
      <div style={{
        position: 'sticky',
        top: '0',
        zIndex: '1000',
        display: 'flex',
        justifyContent: 'flex-end',
        padding: '10px',
        backgroundColor: 'white',
        borderBottom: '1px solid #eee',
        marginBottom: '20px'
      }}>
        <TranslationToggle
          chapterId={metadata.unversionedId || metadata.id}
          content={contentForTranslation}
          onContentChange={() => {}}
        />
      </div>

      {/* Main content */}
      <div>
        {children}
      </div>

      {/* Chatbot - Bottom right */}
      <div style={{ position: 'relative' }}>
        <Chatbot />
      </div>
    </div>
  );
};

export default LayoutWrapper;