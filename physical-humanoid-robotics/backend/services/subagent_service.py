from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from core.config import settings
from models.content import SubagentExecutionLog
from datetime import datetime
import json
from agents.robotics_explainer_agent import RoboticsExplainerAgent
from agents.ros2_code_agent import ROS2CodeAgent
from agents.urdu_translator_agent import UrduTranslatorAgent
from agents.personalization_agent import PersonalizationAgent

class SubagentService:
    def __init__(self, db: Session):
        self.db = db
        self.robotics_agent = RoboticsExplainerAgent()
        self.ros2_agent = ROS2CodeAgent()
        self.urdu_agent = UrduTranslatorAgent()
        self.personalization_agent = PersonalizationAgent()

    def execute_robotics_explainer_agent(self, query: str, user_background: str = None) -> Dict[str, Any]:
        """
        Execute the robotics content explanations subagent.
        """
        # Log the execution
        self._log_subagent_execution("RoboticsExplainerAgent", {"query": query, "user_background": user_background})

        # Use the agent
        result = self.robotics_agent.explain_concept(query, user_background)

        return result

    def execute_ros2_code_agent(self, task_description: str) -> Dict[str, Any]:
        """
        Execute the ROS2 code generation subagent.
        """
        # Log the execution
        self._log_subagent_execution("ROS2CodeAgent", {"task_description": task_description})

        # Use the agent
        result = self.ros2_agent.generate_code(task_description)

        return result

    def execute_urdu_translator_agent(self, text: str, context: str = None) -> Dict[str, Any]:
        """
        Execute the Urdu translation subagent.
        """
        # Log the execution
        self._log_subagent_execution("UrduTranslatorAgent", {"text": text[:50] + "...", "context": context})

        # Use the agent
        result = self.urdu_agent.translate_to_urdu(text, context)

        return result

    def execute_personalization_agent(self, content: str, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the personalization subagent to adapt content based on user preferences.
        """
        # Log the execution
        self._log_subagent_execution("PersonalizationAgent", {
            "content_length": len(content),
            "user_preferences": user_preferences
        })

        # Use the agent
        result = self.personalization_agent.suggest_content_adaptation(content, user_preferences)

        return result

    def _log_subagent_execution(self, subagent_name: str, input_params: Dict[str, Any], user_id: str = None):
        """
        Log the execution of a subagent.
        """
        log_entry = SubagentExecutionLog(
            subagent_name=subagent_name,
            input_params=json.dumps(input_params),
            output_result="",  # This would be filled when we have the result
            execution_time=datetime.utcnow(),
            user_id=user_id
        )

        self.db.add(log_entry)
        self.db.commit()