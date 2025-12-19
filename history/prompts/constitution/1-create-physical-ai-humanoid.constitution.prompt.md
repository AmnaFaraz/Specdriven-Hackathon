---
id: 1
title: Create Physical AI Humanoid Constitution
stage: constitution
date: 2025-12-18
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: Assistant
command: /sp.constitution
labels: [constitution,spec-driven-development,ai-book,physical-ai,humanoid-robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - none
---

## Prompt

```text
You are an expert AI software architect, technical author, and spec-driven development specialist.

Create the sp.constitution for a unified AI-driven book project built using Spec-Kit Plus and Claude Code.

PROJECT NAME:
Physical AI & Humanoid Robotics – Spec-Driven Interactive Book

CORE VISION:
This project delivers a high-quality, production-grade interactive technical book focused on Physical AI and Humanoid Robotics. The book bridges digital intelligence and embodied intelligence using modern AI tooling, robotics platforms, and spec-driven engineering practices.

NON-NEGOTIABLE PRINCIPLES:
1. Spec-Driven Development First
   - Every feature must originate from a clear specification before implementation.
   - No ad-hoc coding or undocumented logic.
   - Specs are the single source of truth.

2. AI-Augmented Authoring
   - Claude Code and Spec-Kit Plus must be used for structured content generation.
   - Reusable intelligence (subagents and agent skills) should be supported and encouraged.

3. Modular, Maintainable Architecture
   - Frontend, backend, RAG pipeline, authentication, and personalization must be decoupled.
   - Clear boundaries between content, UI, AI logic, and infrastructure.

4. Production-Grade Quality
   - Clean code, clear documentation, and deployable artifacts.
   - No demo-only shortcuts.

5. Human-Centered Design
   - The book must be readable, interactive, and visually engaging.
   - Accessibility and personalization are first-class concerns.

TECH STACK (MANDATORY):
Frontend & Content:
- Docusaurus for book generation
- Custom purplish gradient theme (modern, minimal, attractive)
- Deployed on GitHub Pages (primary) and Vercel (secondary)

Backend & AI:
- FastAPI for backend services
- OpenAI Agents / ChatKit SDKs for conversational AI
- Retrieval-Augmented Generation (RAG) architecture

Databases & Vector Search:
- Neon Serverless Postgres for user data, auth state, and chat history
- Qdrant Cloud (Free Tier) for vector embeddings and retrieval

Authentication & Personalization:
- better-auth.com for Signup & Signin
- Collect user software and hardware background at signup
- Personalize chapter content based on user background
- Allow per-chapter personalization toggle
- Allow per-chapter Urdu translation toggle

RAG CHATBOT REQUIREMENTS:
- Embedded directly inside the Docusaurus book UI
- Can answer questions about:
  a) Entire book
  b) Current chapter
  c) Only user-selected text
- Maintains conversation history per user
- Clearly cites retrieved sources

DEPLOYMENT & INFRASTRUCTURE:
- GitHub as source of truth
- Vercel deployment for frontend and backend
- Environment variables securely managed
- CI-friendly project structure

DATA & HISTORY:
- All chatbot interactions must be stored
- User personalization choices must persist
- Clear schema definitions required

QUALITY BAR:
- Clear learning progression aligned with the course outline
- Accurate technical explanations (ROS 2, Gazebo, Isaac, VLA)
- No hallucinated APIs or tools
- Consistent terminology and formatting

OUT OF SCOPE:
- No mobile apps
- No closed-source dependencies that block deployment
- No unverified robotics claims

SUCCESS DEFINITION:
A visually polished, AI-powered, spec-driven interactive book that teaches Physical AI & Humanoid Robotics, deploys successfully, and demonstrates advanced AI engineering practices.

Write this constitution clearly, concisely, and formally.
```

## Response snapshot

Created a comprehensive constitution document for the Physical AI & Humanoid Robotics interactive book project. The constitution includes six core principles: Spec-Driven Development First, AI-Augmented Authoring, Modular Architecture, Production-Grade Quality, Human-Centered Design, and Technical Accuracy. It also details technology stack requirements, implementation guidelines, development workflows, quality standards, and governance procedures. The document was written to the file `.specify/memory/constitution.md` and follows all specified requirements for the interactive book project.

## Outcome

- ✅ Impact: Created initial project constitution document with all specified principles and requirements
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with comprehensive constitution
- 🔁 Next prompts: Development can now proceed following the established constitutional principles
- 🧠 Reflection: The constitution establishes a solid foundation for spec-driven development of the AI-powered interactive book project

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - Constitution meets all specified requirements
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin creating feature specifications based on the constitution