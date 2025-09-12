from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from final_smart_ai import FinalSmartMotorcycleAI
import uuid
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Smart Motorcycle AI API",
    description="API untuk konsultasi masalah motor dengan AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI instance
ai = FinalSmartMotorcycleAI()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    confidence: Optional[float] = None

class SessionSummaryResponse(BaseModel):
    summary: str
    session_id: str
    total_interactions: int

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str

# Store active sessions (in production, use Redis or database)
active_sessions: Dict[str, Dict[str, Any]] = {}

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Smart Motorcycle AI API is running",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        message="All systems operational",
        timestamp=datetime.now().isoformat()
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Main chat endpoint untuk konsultasi motor"""
    try:
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
        
        # Validate input
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get AI response
        response = ai.diagnose(request.message.strip(), request.session_id)
        
        # Track session
        if request.session_id not in active_sessions:
            active_sessions[request.session_id] = {
                "created_at": datetime.now().isoformat(),
                "interactions": 0
            }
        
        active_sessions[request.session_id]["interactions"] += 1
        active_sessions[request.session_id]["last_activity"] = datetime.now().isoformat()
        
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/chat/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(session_id: str):
    """Get summary of a chat session"""
    try:
        # Get summary from AI
        summary = ai.get_smart_summary(session_id)
        
        # Get interaction count
        interactions = 0
        if session_id in active_sessions:
            interactions = active_sessions[session_id].get("interactions", 0)
        
        return SessionSummaryResponse(
            summary=summary,
            session_id=session_id,
            total_interactions=interactions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")

@app.get("/sessions")
async def get_active_sessions():
    """Get list of active sessions"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": {
            session_id: {
                "interactions": data.get("interactions", 0),
                "created_at": data.get("created_at"),
                "last_activity": data.get("last_activity")
            }
            for session_id, data in active_sessions.items()
        }
    }

@app.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific chat session"""
    try:
        # Clear from AI memory
        if hasattr(ai, 'sessions') and session_id in ai.sessions:
            del ai.sessions[session_id]
        
        # Clear from active sessions
        if session_id in active_sessions:
            del active_sessions[session_id]
        
        return {"message": f"Session {session_id} cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@app.post("/chat/new")
async def create_new_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        "created_at": datetime.now().isoformat(),
        "interactions": 0
    }
    
    return {
        "session_id": session_id,
        "message": "New session created",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting Smart Motorcycle AI API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 Health Check: http://localhost:8000/health")
    print("💬 Chat Endpoint: POST http://localhost:8000/chat")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )