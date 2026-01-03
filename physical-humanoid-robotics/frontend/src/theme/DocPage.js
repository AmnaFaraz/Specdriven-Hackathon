import React from 'react';
import OriginalDocPage from '@theme-original/DocPage';
import TranslationToggle from '@site/src/components/Translation/TranslationToggle';

export default function DocPage(props) {
  const { content } = props;
  const { metadata } = content;

  // Extract content for translation
  const contentForTranslation = metadata?.title ? `${metadata.title}\n\n${metadata.description || ''}` : '';

  return (
    <>
      {/* Translation Toggle - positioned at the top of the page */}
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
          onContentChange={() => {}} // Content change would require more complex implementation
        />
      </div>

      <OriginalDocPage {...props} />
    </>
  );
}