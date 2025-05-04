#!/usr/bin/env python3
"""
Wrapper script to run the multiagent system and pipe output to the conversational bot.
This script ensures both files are in place and handles configuration.
"""

import os
import sys
import argparse
import subprocess
import time
from dotenv import load_dotenv

load_dotenv()

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import pandas as pd
        import google.generativeai as genai
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages with: pip install pandas google-generativeai")
        return False

def check_api_key():
    """Check if GEMINI_API_KEY is set in the environment"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("Please set this variable by running:")
        print("  export GEMINI_API_KEY=your_api_key")
        return False
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = ["employee_data.csv", "project_data.csv", "financial_data.csv"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"ERROR: The following required data files are missing: {', '.join(missing_files)}")
        print("Please make sure these files are in the current directory.")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Multiagent Project Management System")
    parser.add_argument("--project-id", type=str, default="PRJ001", 
                       help="Project ID to analyze (default: PRJ001)")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive conversation after analysis")
    parser.add_argument("--output-file", type=str,
                       help="Save analysis to output file")
    parser.add_argument("--skip-bot", action="store_true",
                       help="Skip conversational bot and only run analysis")
    
    args = parser.parse_args()
    
    # Check dependencies and configuration
    if not all([check_dependencies(), check_api_key(), check_data_files()]):
        sys.exit(1)
    
    # Determine pipe mode
    pipe_mode = "none" if args.skip_bot else "direct" if args.interactive else "file"
    
    # Build command for multiagent system
    multiagent_cmd = [
        "python", "agent.py",
        "--project-id", args.project_id,
        "--pipe-mode", pipe_mode
    ]
    
    if args.output_file:
        multiagent_cmd.extend(["--output-file", args.output_file])
    
    # Run the multiagent system
    print("\n===== Starting Multiagent Analysis System =====")
    start_time = time.time()
    
    try:
        subprocess.run(multiagent_cmd, check=True)
        
        end_time = time.time()
        print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running multiagent system: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()