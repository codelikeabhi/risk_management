# AgentVerse: AI-Powered Project Management System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentversebycorpusbound.streamlit.app/)

## Overview

AgentVerse is a sophisticated multi-agent system designed to revolutionize project management through AI-driven analytics. It leverages multiple specialized AI agents to analyze projects from different perspectives, providing comprehensive risk assessment, timeline tracking, financial monitoring, and market analysis.

## Live Demo

- **Frontend**: [https://agentversebycorpusbound.streamlit.app/](https://agentversebycorpusbound.streamlit.app/)
- **Backend API**: [https://agentverse-uz89.onrender.com](https://agentverse-uz89.onrender.com)

## Github Repositories

- **Frontend**: [https://github.com/codelikeabhi/AgentVerse](https://github.com/codelikeabhi/AgentVerse)
- **Backend**: [https://github.com/codelikeabhi/risk_management](https://github.com/codelikeabhi/risk_management)

## Features

- **Multi-Agent Architecture**: Specialized AI agents working together to provide holistic analysis
- **Project Risk Assessment**: Analyze employee performance and identify potential risks
- **Timeline Tracking**: Monitor project progress against deadlines
- **Financial Analysis**: Track spending against budgets and analyze cost variances
- **Market Insights**: Monitor market trends, stock performance, and news for relevant companies
- **Conversational Interface**: Interact with the system through natural language
- **MongoDB Integration**: Store and retrieve chat history and project data

## Architecture

AgentVerse consists of four specialized agents coordinated by a master agent:

1. **Employee Risk Agent**: Evaluates employee records and analyzes potential risks or issues
2. **Project Tracking Agent**: Monitors project progress and compares it with expected timelines
3. **Financial Agent**: Tracks project spending and analyzes cost variances
4. **Market Analysis Agent**: Analyzes market trends, stock performance, and news sentiment
5. **Master Agent**: Integrates insights from all specialized agents to provide holistic analysis

## Tech Stack

### Backend
- Python FastAPI for REST API
- MongoDB for data storage
- Google Generative AI (Gemini) for agent intelligence
- yfinance for stock data
- NewsAPI for market news
- Motor for asynchronous MongoDB operations

### Frontend
- Streamlit for user interface
- HTTP requests to interact with the backend API

## Getting Started

### Prerequisites

- Python 3.12+
- MongoDB
- Google Generative AI API key
- NewsAPI key

### Environment Variables

Create a `.env` file with the following variables:

```
GEMINI_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_newsapi_key
MONGODB_URL=your_mongodb_connection_string
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/codelikeabhi/agentverse.git
cd agentverse
```

2. Install backend dependencies
```bash
pip install -r requirements.txt
```

3. Run the backend server
```bash
uvicorn main:app --reload
```

4. In a separate terminal, run the Streamlit frontend
```bash
streamlit run streamlit_app.py
```

## API Endpoints

- `POST /projects/`: Create a new project
- `GET /projects/`: Get all projects
- `GET /chats/{project_id}`: Get chat history for a project
- `POST /chat/init/{project_id}`: Initialize a chat session with CSV data uploads
- `POST /chat/continue/{project_id}`: Continue a chat conversation
- `GET /ping`: Health check endpoint

## Data Format

The system requires three CSV files to initialize a project:

1. **Employee Data**: Information about team members
2. **Project Data**: Details about projects including timelines and milestones
3. **Financial Data**: Financial records related to projects

## Usage

1. Create a new project from the frontend
2. Upload the required CSV files to initialize the project
3. View the AI's initial analysis of your project data
4. Ask questions through the chat interface to get insights about:
   - Project risks and timelines
   - Team performance and issues
   - Financial status and projections
   - Market conditions affecting the project

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Generative AI for powering the agent intelligence
- Streamlit for the interactive frontend framework
- MongoDB for database functionality
- Render and Streamlit Cloud for hosting services
