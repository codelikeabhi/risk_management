import os
import json
import time
import google.generativeai as genai
from typing import Dict, Any, List, Optional
import argparse
import re
from dotenv import load_dotenv

load_dotenv()

class ConversationalBot:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """Initialize the conversational bot with Gemini"""
        self.api_key = api_key
        self.model_name = model_name
        # Initialize the Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.chat = self.model.start_chat(history=[])
        self.original_data = None  # Will store the original data
        
        # Set system prompt for the bot
        self.system_prompt = """
        You are a helpful Project Management Assistant that provides insights based on 
        data from specialized agents that analyze employees, projects, and finances.
        
        When responding to project analysis data:
        1. Summarize the key findings in a concise manner
        2. Showcase all the relevant data points and insights
        3. Highlight the most critical issues that need attention
        4. Provide actionable recommendations based on the analysis
        5. Be conversational and helpful in your responses
        6. Ask relevant follow-up questions to better understand the user's needs

        You have access to the original source data from:
        - Employee records (skills, performance ratings, attendance records, etc.)
        - Project data (milestones, progress, team members, etc.)
        - Financial records (budget, expenses, categories, etc.)
        - Market data (stock prices, news articles, etc.)

        You can use this raw data to provide detailed answers to specific questions or to validate your analysis.
        
        We have used the latest market news and the last 1 month stock prices of both the companies as external real time data to help with the analysis.
        
        Your goal is to make complex project management data accessible and actionable 
        for managers and team leaders.
        """
        
        # Send system prompt
        self._initialize_chat()
    
    def _initialize_chat(self):
        """Initialize chat with system prompt"""
        self.chat.send_message(self.system_prompt)
    
    def process_analysis(self, analysis_data: str, original_data: Dict = None, max_length: int = 8000) -> str:
        """
        Process the analysis data from the multiagent system.
        Breaks large inputs into manageable chunks if needed.
        
        Parameters:
        - analysis_data: The text analysis from the master agent
        - original_data: Dictionary containing all original data sources (optional)
        - max_length: Maximum chunk size for processing
        """
        # Store the original data for future reference
        self.original_data = original_data
        print("------lol---------")
        print(original_data)
        
        # Check if we need to chunk the input
        if len(analysis_data) <= max_length:
            if original_data:
                # First, inform about available data sources
                data_info = self._get_data_overview(original_data)
                self.chat.send_message(f"I have access to the following data sources that I can use to answer your questions:\n\n{data_info}")
            
            # Then send the analysis
            prompt = f"Here is the latest project analysis from our system. Please help me understand the key insights and what actions I should take:\n\n{analysis_data}"
            response = self.chat.send_message(prompt)
            return response.text
        else:
            # Break into chunks and process sequentially
            chunks = self._chunk_text(analysis_data, max_length)
            
            # First, inform about available data sources if we have them
            if original_data:
                data_info = self._get_data_overview(original_data)
                self.chat.send_message(f"I have access to the following data sources that I can use to answer your questions:\n\n{data_info}")
            
            # Send first chunk with introduction
            prompt = f"I'm sending you a large project analysis in multiple parts. Here's part 1 of {len(chunks)}:\n\n{chunks[0]}"
            self.chat.send_message(prompt)
            
            # Send middle chunks
            for i, chunk in enumerate(chunks[1:-1], 2):
                prompt = f"Here's part {i} of {len(chunks)}:\n\n{chunk}"
                self.chat.send_message(prompt)
            
            # Send final chunk and request analysis
            prompt = f"Here's the final part {len(chunks)} of {len(chunks)}:\n\nNow that you have the complete analysis, please help me understand the key insights and what actions I should take."
            response = self.chat.send_message(prompt)
            return response.text
    
    def _get_data_overview(self, data: Dict) -> str:
        """Generate an overview of available data sources"""
        overview = []
        
        # Employee data overview
        if "employees" in data and data["employees"]:
            num_employees = len(data["employees"])
            departments = set(emp.get("department", "Unknown") for emp in data["employees"])
            overview.append(f"- Employee data for {num_employees} employees across {len(departments)} departments")
        
        # Project data overview
        if "projects" in data and data["projects"]:
            num_projects = len(data["projects"])
            overview.append(f"- Project data for {num_projects} projects including progress, milestones, and team members")
        
        # Financial data overview
        if "financial_records" in data and data["financial_records"]:
            num_records = len(data["financial_records"])
            overview.append(f"- Financial records with {num_records} transactions")
        
        # Market data overview
        if "market_data" in data and data["market_data"]:
            num_companies = len(data["market_data"])
            overview.append(f"- Market data for {num_companies} companies including stock prices and news articles")
        
        return "\n".join(overview)
    
    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks of maximum length"""
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs to maintain context
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max length, save chunk and start new one
            if len(current_chunk) + len(paragraph) + 2 > max_length:
                chunks.append(current_chunk)
                current_chunk = paragraph + "\n\n"
            else:
                current_chunk += paragraph + "\n\n"
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _lookup_data_by_query(self, query: str) -> str:
        """Look up specific information from original data based on query"""
        if not self.original_data:
            return "I don't have access to the original data to answer this question in detail."
        
        # Process the query to identify what data to look up
        query = query.lower()
        response = []
        
        # Look up employee information
        if any(term in query for term in ["employee", "team member", "staff", "developer"]):
            employee_id = None
            employee_name = None
            
            # Try to extract employee ID or name from query
            id_match = re.search(r'employee\s+id\s*[:#]?\s*(\w+)', query)
            if id_match:
                employee_id = id_match.group(1)
            
            name_match = re.search(r'employee\s+name\s*[:#]?\s*([a-zA-Z\s]+)', query)
            if name_match:
                employee_name = name_match.group(1).strip()
            
            # Lookup employee by ID or name
            if employee_id or employee_name:
                for emp in self.original_data.get("employees", []):
                    if (employee_id and emp.get("id") == employee_id) or \
                       (employee_name and employee_name.lower() in emp.get("name", "").lower()):
                        response.append(f"Employee Details for {emp.get('name')}:")
                        response.append(f"- Role: {emp.get('role')}")
                        response.append(f"- Department: {emp.get('department')}")
                        response.append(f"- Skills: {', '.join(emp.get('skills', []))}")
                        # Add more employee details as needed
                        break
        
        # Look up project information
        if any(term in query for term in ["project", "deadline", "milestone"]):
            project_id = None
            project_name = None
            
            # Try to extract project ID or name from query
            id_match = re.search(r'project\s+id\s*[:#]?\s*(\w+)', query)
            if id_match:
                project_id = id_match.group(1)
            
            name_match = re.search(r'project\s+name\s*[:#]?\s*([a-zA-Z\s]+)', query)
            if name_match:
                project_name = name_match.group(1).strip()
            
            # Lookup project by ID or name
            if project_id or project_name:
                for proj in self.original_data.get("projects", []):
                    if (project_id and proj.get("id") == project_id) or \
                       (project_name and project_name.lower() in proj.get("name", "").lower()):
                        response.append(f"Project Details for {proj.get('name')}:")
                        response.append(f"- Description: {proj.get('description')}")
                        response.append(f"- Deadline: {proj.get('deadline')}")
                        response.append(f"- Current Progress: {proj.get('current_progress')*100:.1f}%")
                        response.append(f"- Budget: ${proj.get('budget'):,.2f}")
                        response.append(f"- Actual Spend: ${proj.get('actual_spend'):,.2f}")
                        # Add more project details as needed
                        break
        
        # Format and return the response
        if response:
            return "\n".join(response)
        else:
            return "I couldn't find specific information about that in the original data."
    
    def chat_loop(self, original_data: Dict = None):
        """Interactive chat loop for communication with the user"""
        # Store original data if provided
        if original_data:
            self.original_data = original_data
        
        print("\n==== Project Management Assistant ====")
        print("Type 'exit' or 'quit' to end the conversation.")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Assistant: Goodbye! Have a great day.")
                break
            
            # Special handling for detailed data lookup requests
            if any(term in user_input.lower() for term in ["lookup", "details", "specific", "tell me more about"]):
                data_response = self._lookup_data_by_query(user_input)
                if data_response:
                    # Add data lookup results to the user query for context
                    enhanced_input = f"{user_input}\n\nHere is the relevant data from our system:\n{data_response}\n\nPlease use this information in your response."
                    response = self.chat.send_message(enhanced_input)
                else:
                    response = self.chat.send_message(user_input)
            else:
                response = self.chat.send_message(user_input)
            
            print(f"\nAssistant: {response.text}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Conversational Gemini Bot for Project Management")
    parser.add_argument("--interactive", action="store_true", help="Start in interactive mode")
    parser.add_argument("--input-file", type=str, help="File containing analysis data to process")
    parser.add_argument("--data-file", type=str, help="JSON file containing original data sources")
    
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        exit(1)
    
    # Create the bot
    bot = ConversationalBot(api_key)
    
    # Load original data if provided
    original_data = None
    if args.data_file:
        try:
            with open(args.data_file, 'r') as f:
                original_data = json.load(f)
            print(f"Loaded original data from {args.data_file}")
        except Exception as e:
            print(f"Error loading original data file: {e}")
    
    # Process input based on arguments
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                analysis_data = f.read()
            
            response = bot.process_analysis(analysis_data, original_data)
            print("\n==== Project Analysis Response ====")
            print(response)
            
            # Enter interactive mode after processing the file if requested
            if args.interactive:
                bot.chat_loop(original_data)
                
        except Exception as e:
            print(f"Error processing input file: {e}")
    
    # If no input file or interactive mode was explicitly requested
    elif args.interactive:
        bot.chat_loop(original_data)
    
    # If no valid arguments were provided
    else:
        print("Please provide either --interactive flag or --input-file argument")
        print("Example: python conv.py --input-file analysis.txt")
        print("Example: python conv.py --interactive")
        print("Example: python conv.py --input-file analysis.txt --data-file data.json --interactive")

if __name__ == "__main__":
    main()