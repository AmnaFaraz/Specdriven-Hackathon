from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
from core.database import get_db
from sqlalchemy.orm import Session
from services.subagent_service import SubagentService

router = APIRouter()

# Request models
class SubagentRequest(BaseModel):
    query: str
    agent_type: str  # 'robotics_explainer', 'ros2_code', 'urdu_translator', 'personalization'
    user_preferences: Optional[Dict[str, Any]] = None
    context: Optional[str] = None
    user_id: Optional[str] = None

class SubagentExecuteRequest(BaseModel):
    query: str
    agent_type: str
    user_preferences: Optional[Dict[str, Any]] = None
    context: Optional[str] = None
    user_id: Optional[str] = None

# Response models
class SubagentResponse(BaseModel):
    agent_type: str
    result: Dict[str, Any]
    execution_time: Optional[str] = None

@router.post("/subagent/execute", response_model=SubagentResponse)
async def execute_subagent(request: SubagentExecuteRequest, db: Session = Depends(get_db)):
    """
    Execute a specific subagent based on the agent type.
    """
    try:
        subagent_service = SubagentService(db)

        if request.agent_type == "robotics_explainer":
            result = subagent_service.execute_robotics_explainer_agent(
                request.query,
                request.user_preferences.get("background", None) if request.user_preferences else None
            )
        elif request.agent_type == "ros2_code":
            result = subagent_service.execute_ros2_code_agent(request.query)
        elif request.agent_type == "urdu_translator":
            result = subagent_service.execute_urdu_translator_agent(
                request.query,
                request.context
            )
        elif request.agent_type == "personalization":
            result = subagent_service.execute_personalization_agent(
                request.query,
                request.user_preferences if request.user_preferences else {}
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown agent type: {request.agent_type}. Valid types are: robotics_explainer, ros2_code, urdu_translator, personalization"
            )

        return SubagentResponse(
            agent_type=request.agent_type,
            result=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/subagent/types")
async def get_available_subagents():
    """
    Get list of available subagents.
    """
    return {
        "subagents": [
            "robotics_explainer",
            "ros2_code",
            "urdu_translator",
            "personalization"
        ]
    }