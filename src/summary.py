from ollama import chat
import json
from typing import Dict

class stockSummary:
    def __init__(self, summary: Dict):
        self.summary = summary
        self.message = None
    def generate_summary(self):
        prompt = f"""
        Analyze the following structured stock summary and produce the dashboard insight.

        DATA:
        {json.dumps(self.summary, indent=2)}
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a quantitative equity strategist generating compact technical insights for a trading dashboard. "
                    "Outputs must be behavior-focused, technically descriptive, plain text only, and free of fluff or metric restatement. "
                    "Maximum 200 words. Minimum 100 words. "
                    "Never include explanations, bullet points, headings, or meta commentary. "
                    "Do not use markdown format. Plain text only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = chat(
            model="gemma2:2b",
            messages=messages,
            options={
                "temperature": 0.1,
                "top_p": 0.8,
                "num_predict": 300,
            }
        )

        self.message = response.message.content.rstrip()