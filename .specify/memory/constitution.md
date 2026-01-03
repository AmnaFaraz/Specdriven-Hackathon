<<<<<<< HEAD
# [PROJECT_NAME] Constitution
<!-- Example: Spec Constitution, TaskFlow Constitution, etc. -->

## Core Principles

### [PRINCIPLE_1_NAME]
<!-- Example: I. Library-First -->
[PRINCIPLE_1_DESCRIPTION]
<!-- Example: Every feature starts as a standalone library; Libraries must be self-contained, independently testable, documented; Clear purpose required - no organizational-only libraries -->

### [PRINCIPLE_2_NAME]
<!-- Example: II. CLI Interface -->
[PRINCIPLE_2_DESCRIPTION]
<!-- Example: Every library exposes functionality via CLI; Text in/out protocol: stdin/args → stdout, errors → stderr; Support JSON + human-readable formats -->

### [PRINCIPLE_3_NAME]
<!-- Example: III. Test-First (NON-NEGOTIABLE) -->
[PRINCIPLE_3_DESCRIPTION]
<!-- Example: TDD mandatory: Tests written → User approved → Tests fail → Then implement; Red-Green-Refactor cycle strictly enforced -->

### [PRINCIPLE_4_NAME]
<!-- Example: IV. Integration Testing -->
[PRINCIPLE_4_DESCRIPTION]
<!-- Example: Focus areas requiring integration tests: New library contract tests, Contract changes, Inter-service communication, Shared schemas -->

### [PRINCIPLE_5_NAME]
<!-- Example: V. Observability, VI. Versioning & Breaking Changes, VII. Simplicity -->
[PRINCIPLE_5_DESCRIPTION]
<!-- Example: Text I/O ensures debuggability; Structured logging required; Or: MAJOR.MINOR.BUILD format; Or: Start simple, YAGNI principles -->

### [PRINCIPLE_6_NAME]


[PRINCIPLE__DESCRIPTION]

## [SECTION_2_NAME]
<!-- Example: Additional Constraints, Security Requirements, Performance Standards, etc. -->

[SECTION_2_CONTENT]
<!-- Example: Technology stack requirements, compliance standards, deployment policies, etc. -->

## [SECTION_3_NAME]
<!-- Example: Development Workflow, Review Process, Quality Gates, etc. -->

[SECTION_3_CONTENT]
<!-- Example: Code review requirements, testing gates, deployment approval process, etc. -->

## Governance
<!-- Example: Constitution supersedes all other practices; Amendments require documentation, approval, migration plan -->

[GOVERNANCE_RULES]
<!-- Example: All PRs/reviews must verify compliance; Complexity must be justified; Use [GUIDANCE_FILE] for runtime development guidance -->

**Version**: [CONSTITUTION_VERSION] | **Ratified**: [RATIFICATION_DATE] | **Last Amended**: [LAST_AMENDED_DATE]
<!-- Example: Version: 2.1.1 | Ratified: 2025-06-13 | Last Amended: 2025-07-16 -->
=======
<!--
SYNC IMPACT REPORT:
Version change: 1.0.0 → 1.0.0 (initial creation)
Modified principles: None (new constitution)
Added sections: All sections added as initial constitution
Removed sections: None
Templates requiring updates:
- ✅ .specify/templates/plan-template.md - Verify alignment with new principles
- ✅ .specify/templates/spec-template.md - Verify scope alignment
- ✅ .specify/templates/tasks-template.md - Verify task categorization alignment
- ✅ .specify/templates/commands/*.md - Verify no outdated references
- ⚠ README.md - May need updates to reflect new constitution
Follow-up TODOs: None
-->
# Physical AI & Humanoid Robotics – Spec-Driven Interactive Book Constitution

## Core Principles

### I. Spec-Driven Development First
Every feature must originate from a clear specification before implementation. No ad-hoc coding or undocumented logic. Specs are the single source of truth.

### II. AI-Augmented Authoring
Claude Code and Spec-Kit Plus must be used for structured content generation. Reusable intelligence (subagents and agent skills) should be supported and encouraged.

### III. Modular, Maintainable Architecture
Frontend, backend, RAG pipeline, authentication, and personalization must be decoupled. Clear boundaries between content, UI, AI logic, and infrastructure.

### IV. Production-Grade Quality
Clean code, clear documentation, and deployable artifacts. No demo-only shortcuts. All implementations must meet production standards with proper error handling, logging, and monitoring.

### V. Human-Centered Design
The book must be readable, interactive, and visually engaging. Accessibility and personalization are first-class concerns. User experience must be intuitive and barrier-free.

### VI. Technical Accuracy and Verification
No hallucinated APIs or tools. All technical explanations must be accurate and verified. Consistent terminology and formatting throughout the content. Technical claims must be verifiable and grounded in reality.

## Technology Stack and Implementation Requirements

### Frontend & Content
- Docusaurus for book generation and documentation
- Custom purplish gradient theme (modern, minimal, attractive)
- Deployed on GitHub Pages (primary) and Vercel (secondary)
- Responsive design supporting multiple screen sizes

### Backend & AI Services
- FastAPI for backend services
- OpenAI Agents / ChatKit SDKs for conversational AI
- Retrieval-Augmented Generation (RAG) architecture
- Proper error handling, rate limiting, and security measures

### Data Management
- Neon Serverless Postgres for user data, auth state, and chat history
- Qdrant Cloud (Free Tier) for vector embeddings and retrieval
- Clear schema definitions required for all data models
- All chatbot interactions must be stored for analytics and improvement
- User personalization choices must persist across sessions

### Authentication & Personalization
- Better-auth.com for Signup & Signin
- Collect user software and hardware background at signup
- Personalize chapter content based on user background
- Allow per-chapter personalization toggle
- Allow per-chapter Urdu translation toggle
- Secure handling of user credentials and privacy

### RAG Chatbot Capabilities
- Embedded directly inside the Docusaurus book UI
- Can answer questions about: entire book, current chapter, or user-selected text
- Maintains conversation history per user
- Clearly cites retrieved sources
- Provides contextual and accurate responses based on book content

## Development Workflow and Quality Standards

### Specification Requirements
- Every feature must begin with a clear, detailed specification
- Specifications must include acceptance criteria and test scenarios
- Changes to specifications require formal approval process
- Regular reviews to ensure alignment between specs and implementation

### Code Quality Standards
- Clean, well-documented code with appropriate comments
- Consistent formatting and naming conventions
- Comprehensive unit and integration tests
- Code reviews required for all changes
- Adherence to security best practices

### Deployment and Infrastructure
- GitHub as source of truth
- Vercel deployment for frontend and backend
- Environment variables securely managed
- CI-friendly project structure with automated testing
- Monitoring and alerting for production systems

### Content Quality Standards
- Clear learning progression aligned with the course outline
- Accurate technical explanations (ROS 2, Gazebo, Isaac, VLA)
- Consistent terminology and formatting throughout
- Regular fact-checking and updates to maintain accuracy
- User feedback integration for continuous improvement

## Out of Scope

- No mobile apps (focus remains on web-based experience)
- No closed-source dependencies that block deployment
- No unverified robotics claims or speculative technology
- No third-party proprietary systems that compromise user privacy
- No features that deviate from the core educational mission

## Governance

This constitution serves as the governing document for all development activities related to the Physical AI & Humanoid Robotics interactive book project. All team members must adhere to these principles and requirements. Changes to this constitution require formal approval and documentation of the rationale for any amendments. Regular compliance reviews must be conducted to ensure ongoing adherence to these principles.

All implementations must verify compliance with these principles during code reviews and testing phases. Use this constitution as the primary reference for decision-making when trade-offs or conflicting priorities arise.

**Version**: 1.0.0 | **Ratified**: 2025-12-18 | **Last Amended**: 2025-12-18
>>>>>>> 0959fde799531f61e2f4f0f8682c944ad3633a73
