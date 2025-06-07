import os
import json
import requests
from typing import Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

# ===============================================================================
# OpenAI/ OpenRouter ENVIRONMENT SETUP
# ===============================================================================
load_dotenv()
OPENAI_PROVIDER = "openai"
OPENROUTER_PROVIDER = "openrouter"
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "google/gemma-3-27b-it")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ===============================================================================
# API MODEL CALLING
# ===============================================================================
def call_model_responses(
    analysis_results: dict,
    provider: str = OPENROUTER_PROVIDER,
    model: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 32768, # For OpenRouter model(s) for now, subject to change of course.
    **kwargs
) -> str:
    """
    Sending image and stats to the selected provider (OpenAI or OpenRouter) and get a plain-English description.
    Defaults to OpenRouter (for now).
    """
    if provider not in [OPENAI_PROVIDER, OPENROUTER_PROVIDER]:
        raise ValueError(f"Unknown provider: {provider}")
    
    # OpenAI:
    if provider == OPENAI_PROVIDER:
        if client is None:
            raise ValueError("OpenAI client not initialized or API key missing.")
        model_name = model or OPENAI_DEFAULT_MODEL
        instructions = "You are an archaeologist and remote sensing analyst. Analyze the provided data (image and/or stats), then describe the surface features and interpret the statistics in plain English."
        user_message = {
            "role": "user",
            "content": (
                f"Here are basic stats: {json.dumps(analysis_results['statistics'], indent=2)}.\n"
                "Describe the surface features and interpret the statistics in plain English."
            ),
        }
        try:
            response = client.responses.create(
                model=model_name,
                instructions=instructions,
                input=[user_message],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.output_text.strip()
        except Exception as e:
            return f"[OpenAI API error] {str(e)}"

    # OpenRouter:
    elif provider == OPENROUTER_PROVIDER:
        if requests is None:
            return "[OpenRouter error] 'requests' package not installed."
        if not OPENROUTER_API_KEY:
            return "[OpenRouter error] API key missing. Set OPENROUTER_API_KEY in your environment."
        model_name = model or OPENROUTER_DEFAULT_MODEL
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            #"HTTP-Referer": "https://github.com/yourusername/your-repo",  # Optional
            #"X-Title": "OpenAI-to-Z Challenge"  # Optional
        }
        messages = [
            {
                "role": "system",
                "content": ("You are an archaeologist and remote sensing analyst. Analyze the provided data (image and/or stats), "
                            "then describe the surface features and interpret the statistics in plain English." )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": ("Here are the basic stats about this LiDAR elevation data: "
                                 f"{json.dumps(analysis_results['statistics'], indent=2)}. "
                                 "Please analyze the elevation plot and describe the surface features and interpret the statistics in plain English.")
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{analysis_results['image']}"
                        }
                    }
                ]
            }
        ]
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        try:
            print("Sending request to OpenRouter API...")
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            print("Response received from OpenRouter API.", resp.status_code)
            resp_json = resp.json()
            if resp.status_code != 200:
                # Return the full error message from the API if available:
                return f"[OpenRouter API error] {resp_json.get('error', resp.text)}"
            if "choices" not in resp_json:
                # Log the full response for debugging:
                return f"[OpenRouter API error] Unexpected response format: {json.dumps(resp_json, indent=2)}"
            response_text = resp_json["choices"][0]["message"]["content"].strip()
            print(f"OpenRouter API usage: {resp_json.get('usage', 'unknown')}")
            return response_text
        except Exception as e:
            return f"[OpenRouter API error] {str(e)}"
    else:
        return f"[Provider error] Unknown provider: {provider}"

# ===============================================================================
# BACKWARD-COMPATIBLE OPENAI FUNCTION (optional usage, easy for OpenAI users)
# ===============================================================================
def call_openai_responses(analysis_results: dict, model: str = OPENAI_DEFAULT_MODEL) -> str:
    """
    Backward-compatible: Send `analysis_results` to the OpenAI Responses API and get a plain-English description.
    Equivalent to call_model_responses(..., provider='openai').
    """
    return call_model_responses(analysis_results, provider=OPENAI_PROVIDER, model=model)
