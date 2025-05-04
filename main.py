import os
import tempfile
from datetime import datetime
from typing import List, Optional, Annotated, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, BeforeValidator, ConfigDict
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import pymongo
from fastapi.middleware.cors import CORSMiddleware

# Import the multiagent system functions and loaders from your agent module.
from agent import setup_multiagent_system, load_employee_data, load_project_data, load_financial_data, pipe_to_conversation_bot

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Multiagent Chat Backend with MongoDB")

origins = [
    "http://localhost:8501",         # For local development with Streamlit
    "https://agentversebycorpusbound.streamlit.app/"  # Replace with your actual deployed frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # Allow requests from these origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection details
MONGODB_URL = os.getenv("MONGODB_URL")
if not MONGODB_URL:
    raise ValueError("MONGODB_URL not set in environment variables")

# MongoDB client
client = AsyncIOMotorClient(MONGODB_URL)
db = client.multiagent_chat_db

# Global store for the chat session. This simple solution supports only one session.
chat_session = {}

# Pydantic v2 compatible ObjectId field
def validate_object_id(v: Any) -> ObjectId:
    if isinstance(v, ObjectId):
        return v
    if isinstance(v, str) and ObjectId.is_valid(v):
        return ObjectId(v)
    raise ValueError("Invalid ObjectId")

PyObjectId = Annotated[ObjectId, BeforeValidator(validate_object_id)]

# Function to get next sequence value for project_id
async def get_next_sequence_value(db, sequence_name: str) -> int:
    """Get the next value in a sequence by name"""
    sequence_document = await db.counters.find_one_and_update(
        {"_id": sequence_name},
        {"$inc": {"sequence_value": 1}},
        upsert=True,
        return_document=pymongo.ReturnDocument.AFTER
    )
    return sequence_document["sequence_value"]

# Pydantic models for MongoDB documents
class ProjectModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    project_id: int = Field(...)  # Will be set by the create function
    name: str
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class ChatMessageModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    project_id: int
    message: str = ""
    response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class ProjectOut(BaseModel):
    project_id: int
    name: str
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

# Input models
class ProjectCreate(BaseModel):
    name: str

class ChatMessage(BaseModel):
    text: str

# Database dependency
async def get_db():
    return db

@app.on_event("startup")
async def startup_db_client():
    # Create indexes for faster queries
    await db.projects.create_index([("project_id", pymongo.ASCENDING)], unique=True)
    await db.chats.create_index([("project_id", pymongo.ASCENDING)])
    
    # Initialize counter collection if it doesn't exist
    if not await db.counters.find_one({"_id": "project_id"}):
        await db.counters.insert_one({"_id": "project_id", "sequence_value": 0})

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

@app.post("/projects/", response_model=ProjectOut)
async def create_project(project: ProjectCreate, db=Depends(get_db)):
    """Create a new project with a name and auto-generated incremental ID"""
    # Get the next sequential project ID
    next_id = await get_next_sequence_value(db, "project_id")
    
    # Create project with sequential ID
    project_data = ProjectModel(name=project.name, project_id=next_id)
    project_dict = project_data.model_dump(by_alias=True)
    if project_dict.get("_id") is None:
        project_dict.pop("_id", None)
    
    result = await db.projects.insert_one(project_dict)
    
    # Retrieve the created project
    created_project = await db.projects.find_one({"_id": result.inserted_id})
    return ProjectOut(project_id=created_project["project_id"], name=created_project["name"])

@app.get("/projects/", response_model=List[ProjectOut])
async def get_all_projects(db=Depends(get_db)):
    """Get all projects"""
    projects = []
    async for project in db.projects.find():
        projects.append(ProjectOut(project_id=project["project_id"], name=project["name"]))
    return projects

@app.get("/chats/{project_id}", response_model=List[ChatMessageModel])
async def get_chat_by_project_id(project_id: int, db=Depends(get_db)):
    """Get all chat messages for a specific project"""
    chats = []
    async for chat in db.chats.find({"project_id": project_id}).sort("timestamp", 1):
        chats.append(ChatMessageModel(**chat))
    return chats

@app.post("/chat/init/{project_id}", response_class=PlainTextResponse)
async def init_chat(
    project_id: int,
    employee_file: UploadFile = File(...),
    project_file: UploadFile = File(...),
    financial_file: UploadFile = File(...),
    db=Depends(get_db)
):
    """
    Initialize a chat session by loading the 3 CSV files via file upload.
    The data is loaded using the helper functions from agent.py.
    After loading, it calls the conversational bot to produce an initial analysis response.
    """
    # Check if project exists
    project = await db.projects.find_one({"project_id": project_id})
    if not project:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

    # Ensure the required API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in environment.")

    # Set up the multiagent system
    agents = setup_multiagent_system(api_key)

    # Save each uploaded file to a temporary file so that the load functions can read it.
    try:
        # Save employee CSV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_emp:
            emp_content = await employee_file.read()
            tmp_emp.write(emp_content)
            emp_path = tmp_emp.name

        # Save project CSV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_proj:
            proj_content = await project_file.read()
            tmp_proj.write(proj_content)
            proj_path = tmp_proj.name

        # Save financial CSV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_fin:
            fin_content = await financial_file.read()
            tmp_fin.write(fin_content)
            fin_path = tmp_fin.name

        # Load CSV data into the system
        success_emp = load_employee_data(agents["employee_agent"], emp_path)
        success_proj = load_project_data(agents["project_agent"], proj_path)
        success_fin = load_financial_data(agents["financial_agent"], fin_path)

        # Clean up temporary files
        os.unlink(emp_path)
        os.unlink(proj_path)
        os.unlink(fin_path)

        if not (success_emp and success_proj and success_fin):
            raise HTTPException(status_code=500, detail="Error loading one or more CSV files.")

        # Save the agents instance with project ID reference
        chat_session["agents"] = agents
        chat_session["project_id"] = project_id

        # Pick a project ID to run analysis on (using the first available project from the uploaded data)
        if agents["project_agent"].projects_database:
            analyze_project_id = next(iter(agents["project_agent"].projects_database.keys()))
        else:
            raise HTTPException(status_code=500, detail="No projects found in the database!")

        # Get analysis data from the master agent
        analysis = agents["master_agent"].comprehensive_analysis(analyze_project_id)

        # Call pipe_to_conversation_bot in "direct" mode and get its response
        response_text = pipe_to_conversation_bot(analysis, agents, pipe_mode="direct")
        
        # Save the initial chat message (empty input, only response)
        chat_message = ChatMessageModel(
            project_id=project_id,
            message="",  # Empty input for initialization
            response=response_text
        )
        
        chat_dict = chat_message.model_dump(by_alias=True)
        if chat_dict.get("_id") is None:
            chat_dict.pop("_id", None)
            
        await db.chats.insert_one(chat_dict)
        
        return response_text

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception during initialization: {str(e)}")

@app.post("/chat/continue/{project_id}", response_class=PlainTextResponse)
async def continue_chat(
    project_id: int, 
    message: ChatMessage,
    db=Depends(get_db)
):
    """
    Continue the chat session by accepting a text input and returning the AI model's response.
    The text is appended to the existing conversation maintained by the master agent.
    """
    # Check if project exists
    project = await db.projects.find_one({"project_id": project_id})
    if not project:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
        
    # Check if the current session matches the requested project
    if chat_session.get("project_id") != project_id:
        raise HTTPException(status_code=400, detail="Chat session doesn't match the requested project ID")
        
    agents = chat_session.get("agents")
    if not agents:
        raise HTTPException(status_code=400, detail="No chat session initialized. Please call /chat/init first.")

    # Use the master agent to process the incoming text.
    master_agent = agents["master_agent"]
    try:
        response_text = master_agent.process(message.text)
        
        # Save the chat message and response
        chat_message = ChatMessageModel(
            project_id=project_id,
            message=message.text,
            response=response_text
        )
        
        chat_dict = chat_message.model_dump(by_alias=True)
        if chat_dict.get("_id") is None:
            chat_dict.pop("_id", None)
            
        await db.chats.insert_one(chat_dict)
        
        return response_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/ping", response_class=PlainTextResponse)
async def ping():
    return "OK"