import os, json
from dotenv import load_dotenv
from openai import OpenAI          

# ===============================================================================
# OpenAI ENVIRONMENT SETUP
# ===============================================================================
load_dotenv()                        
client = OpenAI()                    
MODEL_NAME = "gpt-4o"                 # or "o3-8k", "gpt-4o-mini", etc.

# ===============================================================================
# OpenAI RESPONSES API USAGE
# ===============================================================================
def call_openai_responses(stats: dict, model: str = MODEL_NAME) -> str:
    """
    Send `stats` to the OpenAI Responses API and get a plain-English description.
    """
    instructions = "You are an archaeologist."

    user_message = {
        "role": "user",
        "content": (
            f"Here are basic stats: {json.dumps(stats)}.\n"
            "Describe the surface features in plain English."
        ),
    }

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=[user_message],   # could mix in tool calls later
        # any extra args: temperature=0, stream=False, etc.
    )

    return response.output_text.strip() 

# Example usage (do not run if just producing code):
# stats = {"mean_elev": 123.4, "max_elev": 140.9}
# desc = call_openai_responses(stats)
# print(desc)