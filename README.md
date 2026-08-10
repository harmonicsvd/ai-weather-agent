# Agentic Tool Backend Service

## 🎯 Project Vision

Create a modular, extensible backend service that handles tool execution for AI agents. Separate conversation management from task execution, allowing the voice assistant to focus on natural language understanding while this service handles API integrations, data processing, and external service interactions securely and efficiently. The system is designed to support multiple tools - currently calendar operations are implemented, but the architecture allows for easy addition of new tools.

## 🎯 What This Does

A backend service that executes tools for the voice assistant. It provides a simple API for calendar operations and directly integrates with Google Calendar API using user OAuth tokens. This service eliminates intermediate API calls by handling Google Calendar operations directly.

## 🔗 How It Connects

The voice assistant calls this backend to execute tools:

```
Voice Agent → POST /internal/tools/{tool_name} → Backend Service → Google Calendar API → Return Results
```

**Connection Details:**
- **Backend URL**: `http://127.0.0.1:9000`
- **Authentication**: Internal API key via `X-Internal-API-Key` header
- **Voice Agent Repo**: `agentic-voice-assistant`
- **Shared Database**: Both services use the same Supabase PostgreSQL instance
- **Google Calendar Integration**: Direct API calls using user OAuth tokens from database

## 🛠️ Available Tools

### **Currently Working:**
- **create_event_tool**: Creates Google Calendar events directly using user OAuth tokens
- **meetings_summary_tool**: Fetches and summarizes calendar events directly from Google Calendar API

### **How Tools Work:**
1. Voice agent sends tool name and parameters to `POST /internal/tools/{tool_name}`
2. Backend fetches user profile (including OAuth refresh token) from shared database
3. Backend executes the tool using LangChain with direct Google Calendar API access
4. Results are returned to the voice agent

### **Extensibility:**
The service is designed for easy addition of new tools:
- **Tool Registration**: Simple decorator-based registration system
- **Centralized Configuration**: Tool-specific configs in `apps/tools/`
- **Generic Execution**: Single endpoint handles all registered tools
- **Type Safety**: Pydantic schemas for input validation
- **Error Handling**: Consistent error responses and logging

## 🏗️ Technology Stack

- **FastAPI**: Modern async web framework with automatic API documentation
- **Python 3.12**: Latest Python with async/await patterns for high performance
- **LangChain**: Tool management and orchestration framework
- **PostgreSQL**: Database hosted on Supabase for persistent storage
- **psycopg[binary]**: PostgreSQL database driver with async support
- **Google Calendar API**: Direct calendar integration using OAuth 2.0
- **Google OAuth 2.0**: Secure user authentication and token management
- **HTTPX**: Async HTTP client for internal service communication
- **Pydantic**: Data validation and settings management
- **Tenacity**: Retry logic for resilient API calls

## 🚀 How to Run

### **Quick Start**
Use the provided startup script to run all services together:
```bash
cd ../
./start-all.sh
```

### **Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables in .env file
```

### **Start Service**
```bash
# Option 1: Start all services at once using the startup script
cd ../
./start-all.sh

# Option 2: Start backend service individually (port 9000)
python -m uvicorn apps.api.main:app --reload
```

### **Environment Variables**
- `DATABASE_URL`: Supabase PostgreSQL connection string (shared with voice agent)
- `GOOGLE_OAUTH_CLIENT_ID`: Google OAuth client ID for Calendar API access
- `GOOGLE_OAUTH_CLIENT_SECRET`: Google OAuth client secret for Calendar API access
- `WEATHER_INTERNAL_API_KEY`: Internal API key for tool execution (must match voice agent's `WEATHER_AGENT_INTERNAL_API_KEY`)
- `PROFILE_API_BASE_URL`: Voice agent URL for profile data (`http://127.0.0.1:8000`)
- `PROFILE_INTERNAL_API_KEY`: Voice agent internal API key (must match voice agent's `INTERNAL_API_KEY`)

### **Database Setup**
- **Provider**: Supabase (PostgreSQL hosting)
- **Shared Instance**: Both voice assistant and backend service use the same database
- **Purpose**: User profiles, OAuth refresh tokens, and application state
- **Setup**: Run the migration scripts in `migrations/` folder to initialize database schema
- **Important**: Database must include `google_refresh_token` column in `user_profiles` table for OAuth authentication

## 📡 API Endpoints

### **Tool Execution**
- `POST /internal/tools/{tool_name}` - Execute a specific tool
- `GET /internal/tools` - List all available tools
- `GET /health` - Health check

### **Profile Data**
- `GET /internal/profile/{sub}` - Internal endpoint for fetching user profiles (used by backend service to get OAuth tokens)

### **Internal Service Communication**
- **Profile API**: Backend service calls voice agent's `/internal/profile/{sub}` to get user data
- **Authentication**: Mutual authentication using shared internal API keys
- **Error Handling**: Retry logic with exponential backoff for resilient communication
- **Timeout**: Configurable timeout for external service calls

### **Authentication**
All endpoints require `X-Internal-API-Key` header for security.

## 🔐 OAuth Token Management

### **Authentication Flow:**
1. User authenticates via voice agent's Google OAuth flow
2. Voice agent stores OAuth refresh token in shared database (`user_profiles` table)
3. Backend service fetches user profile (including refresh token) when executing tools
4. Backend service uses refresh token to obtain access token for Google Calendar API
5. Google Calendar operations performed with user's own credentials

### **Security Notes:**
- OAuth tokens are stored securely in the database
- Refresh tokens are only accessible to authenticated backend services
- Internal API keys prevent unauthorized access to profile data
- Google OAuth scopes are limited to calendar access only

## 🏗️ System Architecture

### **Service Design:**
The backend service follows a clean architecture pattern:

```
┌─────────────────────────────────────────────────────────┐
│              FastAPI Application Layer                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  API Endpoints (/internal/tools/*, /health)     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Tool Execution Layer                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Tool Registry | LangChain Integration          │  │
│  │  create_event_tool | meetings_summary_tool     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              External Integration Layer                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Google Calendar API | Profile Client           │  │
│  │  OAuth Token Management | HTTPX Client           │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Data Layer                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  PostgreSQL (Supabase) | User Profiles           │  │
│  │  OAuth Tokens | Application State               │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### **Request Flow:**
1. Voice agent sends tool execution request with parameters
2. API layer validates authentication and forwards to tool registry
3. Tool registry locates appropriate tool and validates input
4. Tool execution layer fetches user profile (including OAuth tokens)
5. External integration layer calls Google Calendar API with user credentials
6. Results flow back through layers to voice agent
7. All steps include error handling, logging, and retry logic
