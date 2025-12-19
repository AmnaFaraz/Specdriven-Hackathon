# Data Model: Physical AI & Humanoid Robotics Interactive Book

## User Entity
- **user_id** (UUID, primary key)
- **email** (string, unique, required)
- **name** (string, required)
- **password_hash** (string, required)
- **created_at** (timestamp, default: now)
- **updated_at** (timestamp, default: now)
- **software_background** (string, optional)
- **hardware_background** (string, optional)
- **preferences** (JSON, optional)

### Relationships
- One-to-Many: User → ChatHistory
- One-to-Many: User → PersonalizationSettings

## ChatHistory Entity
- **chat_id** (UUID, primary key)
- **user_id** (UUID, foreign key to User)
- **session_id** (string, required)
- **query** (text, required)
- **response** (text, required)
- **source_citations** (JSON, optional)
- **created_at** (timestamp, default: now)
- **chapter_context** (string, optional)

### Relationships
- Many-to-One: ChatHistory → User

## PersonalizationSettings Entity
- **setting_id** (UUID, primary key)
- **user_id** (UUID, foreign key to User)
- **chapter_id** (string, required)
- **difficulty_level** (string, enum: beginner/intermediate/advanced)
- **content_focus** (string, enum: theoretical/practical/application)
- **language_preference** (string, default: en)
- **created_at** (timestamp, default: now)
- **updated_at** (timestamp, default: now)

### Relationships
- Many-to-One: PersonalizationSettings → User

## Chapter Entity
- **chapter_id** (string, primary key)
- **title** (string, required)
- **content** (text, required)
- **content_vector** (vector, for similarity search)
- **metadata** (JSON, optional)
- **created_at** (timestamp, default: now)
- **updated_at** (timestamp, default: now)

## TranslationCache Entity
- **cache_id** (UUID, primary key)
- **original_content_id** (string, required)
- **target_language** (string, required)
- **translated_content** (text, required)
- **created_at** (timestamp, default: now)
- **expires_at** (timestamp, required)

## SubagentExecutionLog Entity
- **log_id** (UUID, primary key)
- **subagent_name** (string, required)
- **input_params** (JSON, required)
- **output_result** (JSON, required)
- **execution_time** (timestamp, default: now)
- **user_id** (UUID, optional, foreign key to User)

## Validation Rules
- User email must be valid email format
- User password must meet security requirements (min 8 chars, mixed case, numbers, special chars)
- ChatHistory query/response must not exceed 10,000 characters
- PersonalizationSettings difficulty_level must be one of allowed values
- Chapter content_vector must be a valid embedding vector
- TranslationCache expires_at must be in the future

## State Transitions
- User registration: PENDING → ACTIVE (after email verification)
- Chat session: ACTIVE → INACTIVE (after 30 minutes of inactivity)
- Personalization setting: DRAFT → ACTIVE (after user confirmation)