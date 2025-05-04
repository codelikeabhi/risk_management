import os
import json
import datetime
import pandas as pd
import numpy as np
import google.generativeai as genai
import subprocess
import tempfile
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

import yfinance as yf
from newsapi import NewsApiClient
import re
from datetime import timedelta

load_dotenv()

# Configuration
class Config:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_name = "gemini-2.0-flash"  # Using Gemini 2.0 Flash model
        # Initialize the Gemini API
        genai.configure(api_key=self.api_key)
    
    def get_model(self):
        return genai.GenerativeModel(self.model_name)

# Base Agent class
class Agent:
    def __init__(self, name: str, config: Config, system_prompt: str):
        self.name = name
        self.config = config
        self.system_prompt = system_prompt
        self.model = config.get_model()
        self.chat_history = []
    
    def process(self, input_data: Any) -> str:
        """Process input data and return a response"""
        formatted_input = self._format_input(input_data)
        
        # For Gemini, we need to use a different approach than the system/user format
        # First, start with just the system prompt
        messages = []
        if self.system_prompt:
            messages.append(self.system_prompt)
        
        # Add previous conversation context
        for msg in self.chat_history:
            messages.append(msg["content"])
        
        # Add the current input
        messages.append(formatted_input)
        
        # Join all messages with separators
        prompt = "\n\n".join(messages)
        
        # Get response directly from the model
        response = self.model.generate_content(prompt)
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": formatted_input})
        self.chat_history.append({"role": "assistant", "content": response.text})
        
        return response.text
    
    def _format_input(self, input_data: Any) -> str:
        if isinstance(input_data, dict):
            # First, convert any pandas objects to dict/list
            processed_data = {}
            for key, value in input_data.items():
                if hasattr(value, 'to_dict'):
                    processed_data[key] = value.to_dict()
                elif isinstance(value, dict):
                    processed_data[key] = self._process_nested_dict(value)
                elif isinstance(value, list):
                    processed_data[key] = self._process_nested_list(value)
                else:
                    processed_data[key] = value
            return json.dumps(processed_data, default=str, indent=2)
        elif isinstance(input_data, list):
            return json.dumps(self._process_nested_list(input_data), default=str, indent=2)
        return str(input_data)

    def _process_nested_dict(self, d):
        result = {}
        for k, v in d.items():
            # Convert tuple keys to strings
            key = str(k) if isinstance(k, tuple) else k
            
            if hasattr(v, 'to_dict'):
                result[key] = v.to_dict()
            elif isinstance(v, dict):
                result[key] = self._process_nested_dict(v)
            elif isinstance(v, list):
                result[key] = self._process_nested_list(v)
            else:
                result[key] = v
        return result

    def _process_nested_list(self, lst):
        result = []
        for item in lst:
            if hasattr(item, 'to_dict'):
                result.append(item.to_dict())
            elif isinstance(item, dict):
                result.append(self._process_nested_dict(item))
            elif isinstance(item, list):
                result.append(self._process_nested_list(item))
            else:
                result.append(item)
        return result
        
    def reset_history(self):
        """Reset the chat history"""
        self.chat_history = []

# Data structures for the specialized agents
@dataclass
class Employee:
    id: str
    name: str
    role: str
    department: str
    join_date: datetime.date
    performance_ratings: List[Dict[str, Any]]  # List of performance reviews/ratings
    projects_history: List[Dict[str, Any]]     # List of projects and contributions
    attendance_record: Dict[str, Any]          # Attendance stats
    skills: List[str]
    certifications: List[Dict[str, Any]]

@dataclass
class Project:
    id: str
    name: str
    description: str
    start_date: datetime.date
    deadline: datetime.date
    parent_company: str
    business_partner: str
    team_members: List[str]  # List of employee IDs
    milestones: List[Dict[str, Any]]
    current_progress: float  # Percentage complete
    status_updates: List[Dict[str, Any]]
    budget: float
    actual_spend: float

@dataclass
class FinancialRecord:
    project_id: str
    date: datetime.date
    category: str
    amount: float
    description: str
    approved_by: str
    budget_category: str


# 1. Employee Risk Factoring Agent
class EmployeeRiskAgent(Agent):
    def __init__(self, config: Config):
        system_prompt = """
        You are an Employee Risk Analysis Agent. Your role is to evaluate employee records 
        and analyze their performance, identifying potential risks or issues. 
        
        You should consider factors such as:
        - Performance trends over time
        - Attendance patterns
        - Project contribution history
        - Skill relevance to current projects
        - Team dynamics and collaboration metrics
        
        Provide a risk score (1-10, where 10 is highest risk) and specific risk factors for each employee.
        When analyzing multiple employees, identify those that may require additional attention.
        Your analysis should be data-driven, balanced, and focused on helping management make informed decisions.
        """
        super().__init__("Employee Risk Agent", config, system_prompt)
        self.employee_database = {}  # Storage for employee records
    
    def add_employee(self, employee: Employee):
        """Add or update an employee record in the database"""
        self.employee_database[employee.id] = employee
    
    def get_employee(self, employee_id: str) -> Optional[Employee]:
        """Retrieve an employee record by ID"""
        return self.employee_database.get(employee_id)
    
    def analyze_employee(self, employee_id: str) -> str:
        """Analyze a specific employee's risk factors"""
        employee = self.get_employee(employee_id)
        if not employee:
            return f"Employee with ID {employee_id} not found."
        
        input_data = {
            "request_type": "single_employee_analysis",
            "employee": {
                "id": employee.id,
                "name": employee.name,
                "role": employee.role,
                "department": employee.department,
                "join_date": str(employee.join_date),
                "performance_ratings": employee.performance_ratings,
                "projects_history": employee.projects_history,
                "attendance_record": employee.attendance_record,
                "skills": employee.skills,
                "certifications": employee.certifications
            }
        }
        
        return self.process(input_data)
    
    def analyze_team(self, team_ids: List[str]) -> str:
        """Analyze risk factors for a team of employees"""
        team_data = []
        for emp_id in team_ids:
            employee = self.get_employee(emp_id)
            if employee:
                team_data.append({
                    "id": employee.id,
                    "name": employee.name,
                    "role": employee.role,
                    "department": employee.department,
                    "join_date": str(employee.join_date),
                    "performance_ratings": employee.performance_ratings,
                    "projects_history": employee.projects_history,
                    "attendance_record": employee.attendance_record,
                    "skills": employee.skills,
                    "certifications": employee.certifications
                })
        
        input_data = {
            "request_type": "team_analysis",
            "team": team_data
        }
        
        return self.process(input_data)
    
    def analyze_department(self, department: str) -> str:
        """Analyze all employees in a specific department"""
        dept_employees = []
        for emp_id, employee in self.employee_database.items():
            if employee.department == department:
                dept_employees.append({
                    "id": employee.id,
                    "name": employee.name,
                    "role": employee.role,
                    "department": employee.department,
                    "join_date": str(employee.join_date),
                    "performance_ratings": employee.performance_ratings,
                    "projects_history": employee.projects_history,
                    "attendance_record": employee.attendance_record,
                    "skills": employee.skills,
                    "certifications": employee.certifications
                })
        
        input_data = {
            "request_type": "department_analysis",
            "department": department,
            "employees": dept_employees
        }
        
        return self.process(input_data)


# 2. Project Tracking Agent
class ProjectTrackingAgent(Agent):
    def __init__(self, config: Config):
        system_prompt = """
        You are a Project Tracking Agent. Your role is to monitor project progress 
        and compare it with expected timelines based on deadlines.
        
        You should:
        - Track milestone completion against projected dates
        - Calculate and analyze schedule variance
        - Identify potential bottlenecks or delays
        - Predict likelihood of on-time delivery
        - Suggest corrective actions when projects fall behind
        
        Provide a project health score (1-10, where 10 is perfectly on track) and highlight 
        critical issues that need attention. Your analysis should be data-driven and focused 
        on helping project managers make informed decisions.
        """
        super().__init__("Project Tracking Agent", config, system_prompt)
        self.projects_database = {}  # Storage for project data
    
    def add_project(self, project: Project):
        """Add or update a project in the database"""
        self.projects_database[project.id] = project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Retrieve a project by ID"""
        return self.projects_database.get(project_id)
    
    def update_project_progress(self, project_id: str, progress: float, status_update: Dict[str, Any]):
        """Update the progress of a project and add a status update"""
        project = self.get_project(project_id)
        if project:
            project.current_progress = progress
            project.status_updates.append(status_update)
    
    def analyze_project(self, project_id: str) -> str:
        """Analyze a specific project's progress against timeline"""
        project = self.get_project(project_id)
        if not project:
            return f"Project with ID {project_id} not found."
        
        # Calculate time metrics
        today = datetime.date.today()
        total_days = (project.deadline - project.start_date).days
        days_passed = (today - project.start_date).days
        days_remaining = (project.deadline - today).days
        
        # Expected progress based on timeline (linear model)
        expected_progress = min(1.0, days_passed / total_days) if total_days > 0 else 0
        
        input_data = {
            "request_type": "project_analysis",
            "project": {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "start_date": str(project.start_date),
                "deadline": str(project.deadline),
                "team_size": len(project.team_members),
                "milestones": project.milestones,
                "current_progress": project.current_progress,
                "status_updates": project.status_updates,
                "budget": project.budget,
                "actual_spend": project.actual_spend
            },
            "timeline_metrics": {
                "total_days": total_days,
                "days_passed": days_passed,
                "days_remaining": days_remaining,
                "expected_progress": expected_progress,
                "progress_variance": project.current_progress - expected_progress
            }
        }
        
        return self.process(input_data)
    
    def analyze_all_projects(self) -> str:
        """Analyze the status of all projects"""
        projects_data = []
        today = datetime.date.today()
        
        for proj_id, project in self.projects_database.items():
            total_days = (project.deadline - project.start_date).days
            days_passed = (today - project.start_date).days
            expected_progress = min(1.0, days_passed / total_days) if total_days > 0 else 0
            
            projects_data.append({
                "id": project.id,
                "name": project.name,
                "deadline": str(project.deadline),
                "days_remaining": (project.deadline - today).days,
                "current_progress": project.current_progress,
                "expected_progress": expected_progress,
                "progress_variance": project.current_progress - expected_progress
            })
        
        input_data = {
            "request_type": "all_projects_analysis",
            "projects": projects_data
        }
        
        return self.process(input_data)


# 3. Financial Agent
class FinancialAgent(Agent):
    def __init__(self, config: Config):
        system_prompt = """
        You are a Financial Analysis Agent. Your role is to track project spending 
        and analyze it based on total budgets and project progress.
        
        You should:
        - Monitor actual expenditure against budgeted amounts
        - Calculate and analyze cost variances
        - Identify spending anomalies or concerning patterns
        - Project final costs based on current spending rates
        - Suggest cost-saving opportunities when appropriate
        
        Provide a financial health score (1-10, where 10 is excellent financial health) and highlight 
        any issues that require attention. Your analysis should be data-driven and focused on 
        helping financial managers make informed decisions.
        """
        super().__init__("Financial Agent", config, system_prompt)
        self.financial_records = []  # List of financial transactions
        self.project_budgets = {}    # Project budgets dictionary
    
    def add_financial_record(self, record: FinancialRecord):
        """Add a financial record to the database"""
        self.financial_records.append(record)
    
    def set_project_budget(self, project_id: str, budget: float):
        """Set or update the budget for a project"""
        self.project_budgets[project_id] = budget
    
    def get_project_spend(self, project_id: str) -> float:
        """Calculate the total spend for a project"""
        return sum(record.amount for record in self.financial_records if record.project_id == project_id)
    
    def analyze_project_finances(self, project_id: str, project_progress: float) -> str:
        """Analyze the financial status of a specific project"""
        project_records = [r for r in self.financial_records if r.project_id == project_id]
        
        if not project_records:
            return f"No financial records found for project ID {project_id}."
        
        budget = self.project_budgets.get(project_id, 0)
        total_spend = sum(record.amount for record in project_records)
        
        # Group spending by category
        category_spend = {}
        for record in project_records:
            category_spend[record.category] = category_spend.get(record.category, 0) + record.amount
        
        # Calculate expected spend based on progress
        expected_spend = budget * project_progress if budget > 0 else 0
        
        input_data = {
            "request_type": "project_financial_analysis",
            "project_id": project_id,
            "financial_data": {
                "budget": budget,
                "total_spend": total_spend,
                "spend_percentage": (total_spend / budget * 100) if budget > 0 else 0,
                "remaining_budget": budget - total_spend,
                "category_breakdown": category_spend,
                "project_progress": project_progress,
                "expected_spend": expected_spend,
                "spend_variance": total_spend - expected_spend,
                "recent_transactions": [
                    {
                        "date": str(r.date),
                        "category": r.category,
                        "amount": r.amount,
                        "description": r.description
                    } 
                    for r in sorted(project_records, key=lambda x: x.date, reverse=True)[:10]
                ]
            }
        }
        
        return self.process(input_data)
    
    def analyze_all_finances(self, projects_progress: Dict[str, float]) -> str:
        """Analyze financial status across all projects"""
        project_financials = []
        
        for project_id in self.project_budgets.keys():
            project_records = [r for r in self.financial_records if r.project_id == project_id]
            budget = self.project_budgets.get(project_id, 0)
            total_spend = sum(record.amount for record in project_records)
            progress = projects_progress.get(project_id, 0)
            expected_spend = budget * progress if budget > 0 else 0
            
            project_financials.append({
                "project_id": project_id,
                "budget": budget,
                "total_spend": total_spend,
                "spend_percentage": (total_spend / budget * 100) if budget > 0 else 0,
                "project_progress": progress,
                "expected_spend": expected_spend,
                "spend_variance": total_spend - expected_spend
            })
        
        input_data = {
            "request_type": "all_projects_financial_analysis",
            "projects_financial_data": project_financials,
            "total_budget": sum(self.project_budgets.values()),
            "total_spend": sum(record.amount for record in self.financial_records)
        }
        
        return self.process(input_data)

# 5. Market Analysis Agent
class MarketAnalysisAgent(Agent):
    def __init__(self, config: Config):
        system_prompt = """
        You are a Market Analysis Agent. Your role is to analyze market trends, stock performance,
        and news sentiment for companies involved in projects.
        
        You should:
        - Track stock price movements and volatility
        - Identify significant news events that may impact project stakeholders
        - Assess market sentiment toward project partners
        - Evaluate financial stability of partner organizations
        - Flag potential market-related risks to projects
        
        Provide a market risk score (1-10, where 10 is highest risk) and summarize key market
        factors that could impact project success. Your analysis should be data-driven and focused 
        on helping project managers understand external market forces.
        """
        super().__init__("Market Analysis Agent", config, system_prompt)
        self.company_to_symbol = {
            # 🇮🇳 Indian IT Companies (NSE)
            "tcs": "TCS.NS",
            "infosys": "INFY.NS",
            "wipro": "WIPRO.NS",
            "hcl": "HCLTECH.NS",
            "tech mahindra": "TECHM.NS",
            "l&t infotech": "LTIM.NS",
            "l&t technology": "LTTS.NS",
            "mindtree": "LTIM.NS",
            "persistent": "PERSISTENT.NS",
            "coforge": "COFORGE.NS",
            "mphasis": "MPHASIS.NS",
            "birlasoft": "BSOFT.NS",
            "zen technologies": "ZENTEC.NS",

            # 🌍 FAANG (US listed)
            "facebook": "META",
            "meta": "META",
            "amazon": "AMZN",
            "apple": "AAPL",
            "netflix": "NFLX",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "microsoft": "MSFT",

            # 💼 Other Global Tech Companies
            "oracle": "ORCL",
            "ibm": "IBM",
            "cisco": "CSCO",
            "sap": "SAP",
            "salesforce": "CRM",
            "adobe": "ADBE",

            # 🏦 Financial Tech Relevant
            "goldman sachs": "GS",
            "jp morgan": "JPM",
            "paypal": "PYPL",
            "visa": "V",
            "mastercard": "MA"
        }
        # NewsAPI key - in a real app, this would come from environment variables or config
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.newsapi = NewsApiClient(api_key=self.news_api_key)
    
    def get_stock_symbol(self, company_name: str) -> str:
        """Get stock symbol for a company name"""
        return self.company_to_symbol.get(company_name.lower())
    
    def get_stock_data(self, company_name: str) -> dict:
        """Get stock data for a company"""
        symbol = self.get_stock_symbol(company_name)
        if not symbol:
            return {"error": f"No stock symbol found for {company_name}"}
        
        try:
            # Define date range (last 30 days)
            end_date = datetime.date.today()
            start_date = end_date - timedelta(days=30)
            
            # Fetch data
            df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                return {"error": f"No stock data found for {company_name} ({symbol})"}
            
            # Calculate key metrics
            current_price = df['Close'].iloc[-1]
            price_30d_ago = df['Close'].iloc[0]
            price_change = ((current_price - price_30d_ago) / price_30d_ago) * 100
            
            # Calculate volatility (standard deviation of daily returns)
            daily_returns = df['Close'].pct_change().dropna()
            volatility = daily_returns.std() * 100
            
            # Get highs and lows
            period_high = df['High'].max()
            period_low = df['Low'].min()
            
            # Calculate trading volume trends
            avg_volume = df['Volume'].mean()
            recent_volume = df['Volume'].iloc[-5:].mean()
            volume_change = ((recent_volume - avg_volume) / avg_volume) * 100
            
            # Convert df to a safe format for JSON serialization
            df_flat = df.reset_index()
            raw_data = []
            for i, row in df_flat.iterrows():
                row_dict = {}
                for col in df_flat.columns:
                    col_key = str(col) if isinstance(col, tuple) else col
                    row_dict[col_key] = row[col]
                raw_data.append(row_dict)
            
            return {
                "symbol": symbol,
                "company": company_name,
                "current_price": current_price,
                "price_change_30d": price_change,
                "volatility_30d": volatility,
                "period_high": period_high,
                "period_low": period_low,
                "avg_volume": avg_volume,
                "recent_volume_change": volume_change,
                "raw_data": raw_data
            }
        except Exception as e:
            return {"error": f"Error fetching stock data for {company_name}: {str(e)}"}
    
    def get_company_news(self, company_name: str, max_articles: int = 10) -> dict:
        """Get news articles for a company"""
        try:
            news_results = self.newsapi.get_everything(
                q=company_name,
                language='en',
                page_size=max_articles,
                page=1
            )
            
            if not news_results.get('articles'):
                return {"error": f"No news found for {company_name}"}
            
            # Process and simplify articles
            processed_articles = []
            for article in news_results.get('articles', []):
                processed_articles.append({
                    "title": article.get('title'),
                    "source": article.get('source', {}).get('name'),
                    "published_at": article.get('publishedAt'),
                    "url": article.get('url'),
                    "description": article.get('description')
                })
            
            return {
                "company": company_name,
                "total_results": news_results.get('totalResults', 0),
                "articles": processed_articles
            }
        except Exception as e:
            return {"error": f"Error fetching news for {company_name}: {str(e)}"}
    
    def analyze_project_market_data(self, project_id: str, parent_company: str, business_partner: str) -> str:
        """Analyze market data for a specific project's related companies"""
        # Get stock data
        parent_stock = self.get_stock_data(parent_company)
        partner_stock = self.get_stock_data(business_partner)
        
        # Get news data
        parent_news = self.get_company_news(parent_company, max_articles=5)
        partner_news = self.get_company_news(business_partner, max_articles=5)
        
        # Prepare input for the model
        input_data = {
            "request_type": "project_market_analysis",
            "project_id": project_id,
            "companies": [
                {
                    "name": parent_company,
                    "relationship": "parent_company",
                    "stock_data": parent_stock,
                    "news": parent_news
                },
                {
                    "name": business_partner,
                    "relationship": "business_partner",
                    "stock_data": partner_stock,
                    "news": partner_news
                }
            ]
        }
        
        return self.process(input_data)
    
    def analyze_all_companies(self, projects_data: dict) -> str:
        """Analyze market data for all companies involved in projects"""
        all_companies = set()
        project_companies = {}
        
        for proj_id, project in projects_data.items():
            parent = project.get('parent_company')
            partner = project.get('business_partner')
            
            if parent:
                all_companies.add(parent)
                if proj_id not in project_companies:
                    project_companies[proj_id] = []
                project_companies[proj_id].append({
                    "name": parent,
                    "relationship": "parent_company"
                })
            
            if partner:
                all_companies.add(partner)
                if proj_id not in project_companies:
                    project_companies[proj_id] = []
                project_companies[proj_id].append({
                    "name": partner,
                    "relationship": "business_partner"
                })
        
        # Get market data for all companies
        company_data = []
        for company in all_companies:
            stock_data = self.get_stock_data(company)
            news_data = self.get_company_news(company, max_articles=3)
            
            company_data.append({
                "name": company,
                "stock_data": stock_data,
                "news": news_data
            })
        
        input_data = {
            "request_type": "all_companies_market_analysis",
            "companies": company_data,
            "project_relationships": project_companies
        }
        
        return self.process(input_data)


# 4. Master Agent (Coordinator)
class MasterAgent(Agent):
    # Update MasterAgent's __init__ to include market_agent
    def __init__(self, config: Config, employee_agent: EmployeeRiskAgent, 
                project_agent: ProjectTrackingAgent, financial_agent: FinancialAgent,
                market_agent: MarketAnalysisAgent):
        system_prompt = """
        You are the Master Coordination Agent for a project management system. Your role is to:
        
        1. Integrate and synthesize information from four specialized agents:
        - Employee Risk Agent: Provides insights on employee performance and risk factors
        - Project Tracking Agent: Monitors project progress against deadlines
        - Financial Agent: Tracks spending against budgets and project progress
        - Market Analysis Agent: Monitors market trends, stock performance, and news for relevant companies
        
        2. Provide holistic analysis that considers:
        - How employee performance is affecting project timelines
        - How project delays might impact financial projections
        - How budget constraints might affect team composition and performance
        - How market conditions and company performance might impact project risks
        
        3. Deliver actionable recommendations that balance:
        - Human resource considerations
        - Project timeline requirements
        - Financial constraints
        - Market and external factors
        
        Your output should be comprehensive yet concise, focusing on the most critical insights 
        and highest-priority recommendations. Present your analysis in a structured format that 
        helps decision-makers quickly understand the current situation and best path forward.
        """
        super().__init__("Master Agent", config, system_prompt)
        self.employee_agent = employee_agent
        self.project_agent = project_agent
        self.financial_agent = financial_agent
        self.market_agent = market_agent

    def comprehensive_analysis(self, project_id: str = None) -> str:
        """Perform a comprehensive analysis using inputs from all agents"""
        # Get project data
        if project_id:
            project_data = self.project_agent.analyze_project(project_id)
            project = self.project_agent.get_project(project_id)
            
            if project:
                # Get team data
                team_analysis = self.employee_agent.analyze_team(project.team_members)
                
                # Get financial data
                financial_analysis = self.financial_agent.analyze_project_finances(
                    project_id, project.current_progress)
                
                # Get market analysis data
                market_analysis = self.market_agent.analyze_project_market_data(
                    project_id, project.parent_company, project.business_partner)
                
                input_data = {
                    "request_type": "comprehensive_project_analysis",
                    "project_id": project_id,
                    "project_name": project.name if project else "Unknown Project",
                    "project_analysis": project_data,
                    "team_analysis": team_analysis,
                    "financial_analysis": financial_analysis,
                    "market_analysis": market_analysis
                }
            else:
                return f"Project with ID {project_id} not found."
        else:
            # Analyze all projects
            projects_progress = {
                proj_id: project.current_progress 
                for proj_id, project in self.project_agent.projects_database.items()
            }
            
            projects_analysis = self.project_agent.analyze_all_projects()
            financial_analysis = self.financial_agent.analyze_all_finances(projects_progress)
            
            # Get department analysis for all relevant departments
            departments = set(emp.department for emp in self.employee_agent.employee_database.values())
            department_analyses = {
                dept: self.employee_agent.analyze_department(dept) for dept in departments
            }
            
            # Get market analysis for all companies
            projects_data = {
                proj_id: {
                    "parent_company": project.parent_company,
                    "business_partner": project.business_partner
                }
                for proj_id, project in self.project_agent.projects_database.items()
            }
            market_analysis = self.market_agent.analyze_all_companies(projects_data)
            
            input_data = {
                "request_type": "comprehensive_organization_analysis",
                "projects_analysis": projects_analysis,
                "financial_analysis": financial_analysis,
                "department_analyses": department_analyses,
                "market_analysis": market_analysis
            }
        
        return self.process(input_data)


# Helper functions for data management
def load_employee_data(employee_agent: EmployeeRiskAgent, csv_file_path: str):
    """Load employee data from a CSV file"""
    try:
        df = pd.read_csv(csv_file_path)
        for _, row in df.iterrows():
            # This is a simplified example - you'd need to adjust based on your actual CSV structure
            employee = Employee(
                id=row['id'],
                name=row['name'],
                role=row['role'],
                department=row['department'],
                join_date=datetime.datetime.strptime(row['join_date'], '%d-%m-%Y').date(),
                performance_ratings=json.loads(row['performance_ratings']),
                projects_history=json.loads(row['projects_history']),
                attendance_record=json.loads(row['attendance_record']),
                skills=row['skills'].split(','),
                certifications=json.loads(row['certifications'])
            )
            employee_agent.add_employee(employee)
        return True
    except Exception as e:
        print(f"Error loading employee data: {e}")
        return False

def load_project_data(project_agent: ProjectTrackingAgent, csv_file_path: str):
    """Load project data from a CSV file"""
    try:
        df = pd.read_csv(csv_file_path)
        for _, row in df.iterrows():
            # This is a simplified example - you'd need to adjust based on your actual CSV structure
            project = Project(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                start_date=datetime.datetime.strptime(row['start_date'], '%d-%m-%Y').date(),
                deadline=datetime.datetime.strptime(row['deadline'], '%d-%m-%Y').date(),
                parent_company=row['parent_company'],
                business_partner=row['business_partner'],
                team_members=row['team_members'].split(','),
                milestones=json.loads(row['milestones']),
                current_progress=float(row['current_progress']),
                status_updates=json.loads(row['status_updates']),
                budget=float(row['budget']),
                actual_spend=float(row['actual_spend'])
            )
            project_agent.add_project(project)
        return True
    except Exception as e:
        print(f"Error loading project data: {e}")
        return False

def load_financial_data(financial_agent: FinancialAgent, csv_file_path: str):
    """Load financial data from a CSV file"""
    try:
        df = pd.read_csv(csv_file_path)
        for _, row in df.iterrows():
            # This is a simplified example - you'd need to adjust based on your actual CSV structure
            record = FinancialRecord(
                project_id=row['project_id'],
                date=datetime.datetime.strptime(row['date'], '%d-%m-%Y').date(),
                category=row['category'],
                amount=float(row['amount']),
                description=row['description'],
                approved_by=row['approved_by'],
                budget_category=row['budget_category']
            )
            financial_agent.add_financial_record(record)
            
            # Update project budget if not already set
            if row['project_id'] not in financial_agent.project_budgets:
                financial_agent.set_project_budget(row['project_id'], float(row['budget']))
        return True
    except Exception as e:
        print(f"Error loading financial data: {e}")
        return False


def pipe_to_conversation_bot(analysis_data: str, agents: dict, pipe_mode: str = "file"):
    """
    Pipe the analysis data to the conversational bot
    
    Parameters:
    - analysis_data: The output from the master agent
    - agents: Dictionary containing all agent objects
    - pipe_mode: How to pipe the data ("file" or "direct")
    """
    # Ensure we have the API key environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable not set. Cannot pipe to conversation bot.")
        return
    
    # Prepare the original data to pass to the conversational bot
    original_data = {
        # Employee data
        "employees": [vars(emp) for emp in agents["employee_agent"].employee_database.values()],
        
        # Project data
        "projects": [vars(proj) for proj in agents["project_agent"].projects_database.values()],
        
        # Financial data
        "financial_records": [vars(record) for record in agents["financial_agent"].financial_records],
        "project_budgets": agents["financial_agent"].project_budgets,
        
        # Market data - Get current market data for all companies in projects
        "market_data": {}
    }
    
    # Add market data for all companies in projects
    for project in agents["project_agent"].projects_database.values():
        if project.parent_company and project.parent_company not in original_data["market_data"]:
            original_data["market_data"][project.parent_company] = {
                "stock_data": agents["market_agent"].get_stock_data(project.parent_company),
                "news": agents["market_agent"].get_company_news(project.parent_company)
            }
        
        if project.business_partner and project.business_partner not in original_data["market_data"]:
            original_data["market_data"][project.business_partner] = {
                "stock_data": agents["market_agent"].get_stock_data(project.business_partner),
                "news": agents["market_agent"].get_company_news(project.business_partner)
            }
    
    if pipe_mode == "file":
        # Create temporary files for both analysis and original data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as analysis_file:
            analysis_file_path = analysis_file.name
            analysis_file.write(analysis_data)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as data_file:
            data_file_path = data_file.name
            # Convert datetime objects to strings for JSON serialization
            original_data_serializable = json.loads(
                json.dumps(original_data, default=lambda obj: 
                    obj.isoformat() if hasattr(obj, 'isoformat') else str(obj))
            )
            json.dump(original_data_serializable, data_file)
        
        try:
            # Call the conversational bot with both temporary files
            subprocess.run([
                "python", "conv.py", 
                "--input-file", analysis_file_path, 
                "--data-file", data_file_path,
                "--interactive"
            ], check=True)
        finally:
            # Clean up the temporary files
            os.unlink(analysis_file_path)
            os.unlink(data_file_path)
            
    elif pipe_mode == "direct":
        # Import the conversational bot directly
        from conv import ConversationalBot
        
        # Initialize the bot with both analysis and original data
        bot = ConversationalBot(api_key)
        
        # Process the analysis with original data
        response = bot.process_analysis(analysis_data, original_data)
        
        # Print the response
        print("\n==== Project Analysis Response ====")
        print(response)
        
        # Start interactive mode
        #bot.chat_loop(original_data)
        return response

def setup_multiagent_system(api_key: str):
    """Set up the multiagent system with all required components"""
    # Initialize configuration
    config = Config(api_key)
    
    # Create agents
    employee_agent = EmployeeRiskAgent(config)
    project_agent = ProjectTrackingAgent(config)
    financial_agent = FinancialAgent(config)
    market_agent = MarketAnalysisAgent(config)
    
    # Create master agent
    master_agent = MasterAgent(config, employee_agent, project_agent, financial_agent, market_agent)
    
    return {
        "employee_agent": employee_agent,
        "project_agent": project_agent,
        "financial_agent": financial_agent,
        "market_agent": market_agent,
        "master_agent": master_agent
    }
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Multiagent Project Management System")
    parser.add_argument("--project-id", type=str, help="Project ID to analyze")
    parser.add_argument("--pipe-mode", choices=["file", "direct", "none"], default="file", 
                       help="How to pipe data to conversational bot (file, direct, or none)")
    parser.add_argument("--export-only", action="store_true", 
                       help="Export analysis to file without piping to conversational bot")
    parser.add_argument("--output-file", type=str, help="Output file for analysis data")
    
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        exit(1)
    
    # Set up the system
    agents = setup_multiagent_system(api_key)
    
    # Load data
    print("Loading employee data...")
    load_employee_data(agents["employee_agent"], "employee_data.csv")
    
    print("Loading project data...")
    load_project_data(agents["project_agent"], "project_data.csv")
    
    print("Loading financial data...")
    load_financial_data(agents["financial_agent"], "financial_data.csv")
    
    # Get analysis
    if args.project_id:
        project_id = args.project_id
    else:
        # Get the first available project ID from the loaded data
        if agents["project_agent"].projects_database:
            project_id = next(iter(agents["project_agent"].projects_database.keys()))
        else:
            print("No projects found in the database!")
            exit(1)
    print(f"Performing comprehensive analysis for project {project_id}...")
    analysis = agents["master_agent"].comprehensive_analysis(project_id)
    
    # Handle output based on args
    if args.export_only or args.output_file:
        output_file = args.output_file or f"analysis_{project_id}.txt"
        with open(output_file, 'w') as f:
            f.write(analysis)
        print(f"Analysis exported to {output_file}")
    
    if not args.export_only:
        if args.pipe_mode == "none":
            # Just print the analysis
            print("\n==== Project Analysis ====")
            print(analysis)
        else:
            # Pipe to conversational bot, now passing the agents dictionary
            print("Piping analysis to conversational bot...")
            pipe_to_conversation_bot(analysis, agents, args.pipe_mode)


# Helper function to set up the multiagent system
