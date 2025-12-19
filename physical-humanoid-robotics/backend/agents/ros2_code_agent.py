from typing import Dict, Any
from core.config import settings
from openai import OpenAI

class ROS2CodeAgent:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate_code(self, task_description: str) -> Dict[str, Any]:
        """
        Generate ROS2 Python code based on the task description.
        """
        prompt = f"""
        Generate Python code for ROS2 based on the following task:
        {task_description}

        The code should include:
        1. Proper ROS2 node structure
        2. Necessary imports
        3. Publisher/subscriber setup if needed
        4. Proper lifecycle management
        5. Error handling
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.5
        )

        code = response.choices[0].message.content

        return {
            "task_description": task_description,
            "generated_code": code
        }