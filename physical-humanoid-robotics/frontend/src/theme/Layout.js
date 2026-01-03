import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import Chatbot from '@site/src/components/Chatbot/Chatbot';

// Layout wrapper that adds the chatbot to all pages
export default function LayoutWrapper(props) {
  return (
    <>
      <OriginalLayout {...props} />
      <Chatbot />
    </>
  );
}