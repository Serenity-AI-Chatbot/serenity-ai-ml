# 🌿 Serenity-AI: Personalized Mental Wellness Companion

## 📖 Project Overview

Serenity AI  is a comprehensive mental health application designed to support users' emotional well-being through journaling, mood tracking, personalized activities, and AI-powered insights.

### 🚀 Key Features
- Personalized mood tracking
- Sentiment-aware journaling
- AI-powered emotional insights
- Activity and exercise recommendations
- Gamification of mental wellness
- Emergency support features


#### Frontend Repository  
https://github.com/Serenity-AI-Chatbot/serenity-ai-frontend

#### Backend Repository  
https://github.com/Serenity-AI-Chatbot/serenity-ai-ml

## 📸 Project Media

### Demo Video
[![Serenity AI Demo](https://img.youtube.com/vi/Pkly11LDIlI/0.jpg)](https://www.youtube.com/watch?v=Pkly11LDIlI)

### Architecture Diagram
![Architecture Diagram](https://github.com/user-attachments/assets/094af6ae-3170-4ecf-9324-f16ce599084a))

### System Flow
![System Flow](https://github.com/user-attachments/assets/f21afabd-40d0-4752-8be0-12ea9f2c2239))

### Screenshots

### Home Page
![Home Page](https://github.com/user-attachments/assets/444e8467-a4ec-4a79-8526-573e322cbe24)
-Interative 3D model of a robot


#### Dashboard
![Dashboard](https://github.com/user-attachments/assets/24c6ff1c-aff9-40c1-a810-cbe1648ad0d5)
- Activity tracking
- Mood trends
- Journal insights
- Recommended activities

#### Journal Interface
![Journal Interface](https://github.com/user-attachments/assets/ee1bea8c-727d-4002-8811-aa272f1eca69)
![Journal Interface](https://github.com/user-attachments/assets/469d47fe-08e0-4fb0-8ddb-0eb84072dd72)
![Journal Interface](https://github.com/user-attachments/assets/0ab65c5b-3f15-44b5-ad03-e0e7795260ce)
![Journal Interface](https://github.com/user-attachments/assets/9e040686-034d-4512-b773-b4a8272ede6d)
- AI-powered journaling
- Emotion detection
- Real-time insights
- Tag suggestions

#### Activity Recommendations
![Activity Recommendations](https://github.com/user-attachments/assets/091e6382-496b-4caf-9cd4-6808b1fc5328)
- Personalized suggestions
- Difficulty levels
- Progress tracking
- Achievement system

### Character Interface
![Character Interface](https://github.com/user-attachments/assets/3f77c98c-c3cf-43cf-9de5-2fe9c5512d0a)
![Character Interface](https://github.com/user-attachments/assets/fb55b0d2-5304-420d-8c29-0d8553c1a91a)
![Character Interface](https://github.com/user-attachments/assets/18f66098-81fd-4111-b504-5c6571145826)
- Interactive Chat Bot which can be used to talk to the user
- with all the context of the user's mood and journal entries



## 🛠 Tech Stack

### Frontend
- Next.js 14
- Tailwind CSS
- Shadcn/UI Components
- Vercel Deployment

### Backend
- FastAPI Service
- Supabase Authentication & Database
- Google Gemini AI (Embeddings & Language Model)

### ML Infrastructure
- Custom ML Model (AWS EC2)
- Spacy & NLTK Models
- Vector Embeddings
- Sentiment Analysis Pipeline

### Database
- Supabase Postgres
- Vector Embeddings
- Row Level Security

## 🔧 Prerequisites

- Python 3.9+
- Node.js 18+
- Supabase Account
- Google AI Studio API Key
- AWS Account (for EC2)
- Vercel Account (optional)

## 📦 Project Structure

```
mental-health-app/
│
├── frontend/                # Next.js Application
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── styles/
│
├── ml-service/             # FastAPI ML Service
│   ├── app/
│   │   ├── main.py
│   │   ├── models/
│   │   │   ├── custom_model.py
│   │   │   └── nlp_models.py
│   │   ├── routers/
│   │   └── utils/
│   └── requirements.txt
│
├── database/                # Supabase Migrations & Functions
│   ├── migrations/
│   └── functions/
│
└── README.md
```

## 🔐 Environment Setup

### 1. Supabase Configuration

1. Create a new Supabase project
2. Enable the `vector` extension
3. Run database migrations
4. Set up Row Level Security (RLS)

#### Required Supabase Environment Variables
```
NEXT_PUBLIC_SUPABASE_URL=your_supabase_project_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
GEMINI_API_KEY=your_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
FASTAPI_API_URL=your_fastapi_api_url
```

### 2. Google Gemini AI Setup

1. Get API key from Google AI Studio
2. Set environment variable

```bash
GEMINI_API_KEY=your_gemini_api_key
```


### 3. Frontend Next.js Setup

```bash
# Install dependencies
cd frontend
npm install

# Run development server
npm run dev
```

### 4. ML Service Setup (AWS EC2)

```bash

# Clone repository and navigate to ml-service
git clone <repository-url>
cd mental-health-app/ml-service

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server with PM2
pm2 start "uvicorn app.main:app --host 0.0.0.0 --port 8000"
```

## 📊 Database Schema

### Core Tables

#### 1. Journals Table
- Stores user journal entries with AI enhancements
- Fields:
  - Basic: id, user_id, title, content
  - AI Insights: summary, mood_tags, embedding, keywords
  - Metadata: created_at, tags
  - Location: nearby_places (JSONB)
  - Related Content: latest_articles (JSONB)

#### 2. Activities Table
- Catalog of wellness activities
- Fields:
  - Basic: id, title, description, category
  - Difficulty: difficulty_level (beginner/intermediate/advanced)
  - Mood Matching: recommended_moods
  - Planning: estimated_duration
  - Organization: tags

#### 3. User Activities Table
- Tracks user engagement with activities
- Fields:
  - References: user_id, activity_id
  - Status: planned_at, completed_at, status
  - Feedback: difficulty_rating, reflection

## 🔍 Database Functions

### 1. get_mood_trends()
sql
Function: get_mood_trends(p_user_id UUID)
Returns: TABLE (entry_date DATE, mood_categories JSONB, total_entries BIGINT)
- Analyzes mood patterns over time
- Groups mood tags by date
- Aggregates mood frequencies
- Provides total entries per day

### 2. get_recommended_activities()
sql
Function: get_recommended_activities(p_user_id UUID, p_current_mood_tags TEXT[])
Returns: TABLE (activity_id UUID, title TEXT, description TEXT, match_score double precision)
- Recommends activities based on current mood
- Calculates mood-activity match scores
- Uses intersection algorithm for matching
- Returns top 5 matched activities

### 3. match_journals()
sql
Function: match_journals(query_embedding VECTOR(768), match_threshold FLOAT, match_count INT, user_id UUID)
Returns: TABLE (id UUID, content TEXT, summary TEXT, created_at TIMESTAMP, similarity FLOAT)
- Performs semantic search on journal entries
- Uses vector embeddings for similarity
- Supports configurable matching threshold
- Returns matched entries with similarity scores

### 4. get_dashboard_insights()
sql
Function: get_dashboard_insights(p_user_id UUID, p_days_back INT DEFAULT 90)
Returns: JSONB
Comprehensive analytics function providing:
- Journal trends:
  - Weekly journal counts
  - Mood distribution
  - Keyword analysis
- Activity trends:
  - Completion rates
  - Difficulty ratings
  - Category distribution
- Summary insights:
  - Total counts
  - Most common moods
  - Activity engagement

### 5. get_journals_by_date()
**Function:** `get_journals_by_date`

**Description:**
Retrieves journal entries for a specific user, optionally filtered by year and month. This function allows users to fetch their journal entries within a particular time frame, making it easier to review and analyze past entries.

**Parameters:**
- `p_user_id UUID`: The unique identifier of the user.
- `p_year INT DEFAULT NULL`: (Optional) The year to filter journal entries.
- `p_month INT DEFAULT NULL`: (Optional) The month to filter journal entries.
- `p_limit INT DEFAULT 50`: (Optional) The maximum number of journal entries to return.

**Returns:**
A table containing the following fields:
- `id UUID`: The unique identifier of the journal entry.
- `title TEXT`: The title of the journal entry.
- `content TEXT`: The main content of the journal entry.
- `summary TEXT`: An AI-generated summary of the journal entry.
- `mood_tags TEXT[]`: Array of mood tags associated with the entry.
- `tags TEXT[]`: Additional tags for categorization.
- `keywords TEXT[]`: Extracted keywords from the entry.
- `song TEXT`: Link to a related song.
- `created_at TIMESTAMP WITH TIME ZONE`: The timestamp when the entry was created.

### 6. get_journal_stats_by_period()
**Function:** `get_journal_stats_by_period`

**Description:**
Fetches journal entries for a user within a specified date range. This function is useful for generating reports and analytics based on user activity over a particular period.

**Parameters:**
- `p_user_id UUID`: The unique identifier of the user.
- `p_start_date DATE`: The start date of the period.
- `p_end_date DATE`: The end date of the period.

**Returns:**
A table containing the following fields:
- `id UUID`: The unique identifier of the journal entry.
- `title TEXT`: The title of the journal entry.
- `content TEXT`: The main content of the journal entry.
- `summary TEXT`: An AI-generated summary of the journal entry.
- `mood_tags TEXT[]`: Array of mood tags associated with the entry.
- `tags TEXT[]`: Additional tags for categorization.
- `keywords TEXT[]`: Extracted keywords from the entry.
- `song TEXT`: Link to a related song.
- `created_at TIMESTAMP WITH TIME ZONE`: The timestamp when the entry was created.

## 🔒 Security Features

### Row Level Security (RLS)
- Enabled on all main tables:
  - journals
  - activities
  - user_activities

### Performance Optimizations
sql
Indexes:
idx_journals_user_id
idx_user_activities_user_id
idx_journals_mood_tags (GIN)
idx_journals_keywords (GIN)
idx_journals_embedding (IVFFLAT)

## 🔧 Vector Support
- Enabled vector extension for semantic search
- Uses 768-dimensional embeddings
- Optimized for similarity searches

## 📈 Data Analysis Features

### Mood Analysis
- Daily and weekly aggregations
- Mood category distribution
- Trend analysis over time

### Activity Tracking
- Completion rates
- Difficulty progression
- Category distribution
- User engagement metrics

### Journal Analytics
- Semantic search capabilities
- Keyword extraction
- Mood pattern recognition
- Content summarization

## 🔄 Integration Points

### AI Services
- Vector embeddings for semantic search
- Mood prediction from journal content
- Content summarization
- Keyword extraction

### External Services
- Location-based recommendations
- Related article suggestions
- Activity recommendations

## 🚀 Performance Considerations

### Optimized Queries
- Uses CTEs for complex analytics
- Efficient date-based grouping
- Indexed text search capabilities
- Vector similarity optimization

### Data Aggregation
- Weekly rollups for trends
- Efficient JSON aggregation
- Optimized mood analysis
- Smart activity matching

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Support

For issues, please open a GitHub issue or contact vishalkamboj9211@gmail.com or dbaheti2003@gmail.com

---

### 🌈 Future Roadmap
- [ ] Multi-language support
- [ ] Advanced machine learning models
- [ ] Integration with health tracking devices
