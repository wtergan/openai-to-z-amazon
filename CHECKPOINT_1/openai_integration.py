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
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "google/gemma-3-27b-it:free")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ===============================================================================
# API MODEL CALLING
# ===============================================================================
def call_model_responses(
    plot_stats: dict,
    provider: str = OPENROUTER_PROVIDER,
    model: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 1000,
    **kwargs
) -> str:
    """
    Send `stats` to the selected provider (OpenAI or OpenRouter) and get a plain-English description.
    Defaults to OpenRouter (for now).
    """
    if provider not in [OPENAI_PROVIDER, OPENROUTER_PROVIDER]:
        raise ValueError(f"Unknown provider: {provider}")
    
    # OpenAI:
    if provider == OPENAI_PROVIDER:
        if client is None:
            raise ValueError("OpenAI client not initialized or API key missing.")
        model_name = model or OPENAI_DEFAULT_MODEL
        instructions = "You are an archaeologist. Analyze the provided statistics and describe the surface features in plain English."
        user_message = {
            "role": "user",
            "content": (
                f"Here are basic stats: {json.dumps(plot_stats)}.\n"
                "Describe the surface features in plain English."
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
                "content": "You are an archaeologist. Analyze the provided plot image and stats, then describe the surface features and stats in plain English."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Here are the basic stats about this LiDAR elevation data: {json.dumps(plot_stats['statistics'], indent=2)}. Please analyze the elevation plot and describe the surface features in plain English."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{plot_stats['plot']}"
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
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            if resp.status_code != 200:
                return f"[OpenRouter API error] {resp.text}"
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[OpenRouter API error] {str(e)}"
    else:
        return f"[Provider error] Unknown provider: {provider}"

# ===============================================================================
# BACKWARD-COMPATIBLE OPENAI FUNCTION (optional usage, easy for OpenAI users)
# ===============================================================================
def call_openai_responses(plot_stats: dict, model: str = OPENAI_DEFAULT_MODEL) -> str:
    """
    Backward-compatible: Send `stats` to the OpenAI Responses API and get a plain-English description.
    Equivalent to call_model_responses(..., provider='openai').
    """
    return call_model_responses(plot_stats, provider=OPENAI_PROVIDER, model=model)
