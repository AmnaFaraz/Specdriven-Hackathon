# Research: Physical AI & Humanoid Robotics Interactive Book

## Decision: Technology Stack Selection
**Rationale**: Selected FastAPI for backend due to its async support, excellent OpenAPI integration, and strong typing capabilities. Docusaurus chosen for frontend due to its excellent documentation capabilities, plugin system, and built-in search functionality. This combination provides a robust foundation for an AI-powered interactive book.

## Decision: Authentication System
**Rationale**: better-auth.com selected as it provides a secure, easy-to-integrate authentication solution that supports social logins and has good documentation. It also handles password reset, session management, and security best practices out-of-the-box.

## Decision: Database Architecture
**Rationale**: Neon Serverless Postgres chosen for its serverless capabilities, PostgreSQL compatibility, and built-in connection pooling. Qdrant Cloud selected for vector storage due to its high performance, similarity search capabilities, and cloud-native architecture, which is ideal for RAG applications.

## Decision: Personalization and Translation Architecture
**Rationale**: Per-chapter personalization and translation will be implemented through API endpoints that return content variations based on user preferences. This allows for dynamic content adaptation without storing multiple versions of each chapter.

## Decision: Claude Code Subagents Design
**Rationale**: Subagents will be designed as specialized skills that can be invoked through the RAG system. Each subagent will handle specific tasks like robotics explanations, ROS2 code generation, and translation, providing reusable intelligence as required by the constitution.

## Decision: Deployment Strategy
**Rationale**: GitHub Pages for primary deployment due to its simplicity and integration with Git workflow. Vercel as secondary deployment for additional features and custom domain support. Backend services will be deployed to Vercel or similar cloud platform.

## Alternatives Considered:
- **Authentication**: Auth0, Firebase Auth, Supabase Auth - better-auth.com chosen for simplicity and self-hosting capability
- **Vector Database**: Pinecone, Weaviate, ChromaDB - Qdrant chosen for its performance and open-source nature
- **Frontend**: Next.js, Gatsby, VuePress - Docusaurus chosen for documentation-focused features
- **Backend**: Django, Flask, Express.js - FastAPI chosen for async support and automatic API documentation