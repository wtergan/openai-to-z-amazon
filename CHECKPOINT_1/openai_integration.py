import os
import json
import requests
from typing import Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

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
# API MODEL USAGE
# ===============================================================================
def call_model_responses(
    analysis_results: dict,
    dataset_type: str,
    provider: str = OPENROUTER_PROVIDER,
    model: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 32768, # For OpenRouter model(s) for now, subject to change of course.
    **kwargs
) -> str:
    """
    Sending analysis results to the selected provider (OpenAI or OpenRouter) and returns a plain-English description.
    Defaults to OpenRouter (for now).
    """
    if provider not in [OPENAI_PROVIDER, OPENROUTER_PROVIDER]:
        raise ValueError(f"Unknown provider: {provider}")
    
    # OpenAI:
    if provider == OPENAI_PROVIDER:
        if client is None:
            raise ValueError("OpenAI client not initialized or API key missing.")
        model_name = model or OPENAI_DEFAULT_MODEL
        
        # Dynamic prompting scheme based on the data type (matching OpenRouter format):
        if dataset_type == 'lidar':
            prompt_intro = ("Here are statistics and a hillshade plot derived from LiDAR elevation data. "
                        "The plot shows elevation on the left and a shaded relief view on the right.")
        else: # sentinel2
            prompt_intro = ("Here are statistics and an RGB thumbnail from a Sentinel-2 satellite median composite. "
                        "The stats include various spectral bands and a calculated NDVI (vegetation index).")
            
        # Conditionally build the content list for the user message; input_text, with input_image if available:
        user_content = [
            {
                "type": "input_text",
                "text": (f"{prompt_intro}\n\n"
                         f"Statistics:\n{json.dumps(analysis_results.get('statistics', {}), indent=2)}\n\n"
                         "Please analyze the provided image and statistics. As an expert archaeologist and remote sensing analyst, "
                         "describe the key features, patterns, and anomalies in the landscape in plain English.")
            }
        ]

        if analysis_results.get("image"):
            user_content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{analysis_results['image']}"
            })
        
        # Instructions for system role/behavior; for response API, this replaces the traditional "system" message role:
        instructions = ("You are an expert archaeologist and remote sensing analyst. Your task is to interpret geospatial data. "
                       "You provide clear, insightful, and concise interpretations based on the data provided.")
        
        user_message = {
            "role": "user",
            "content": user_content
        }
        
        try:
            print(f"Sending request to OpenAI Responses API for model: {model_name}...")
            response = client.responses.create(
                model=model_name,
                instructions=instructions,
                input=[user_message],
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
            print(f"Response received from OpenAI Responses API")
            if not hasattr(response, 'output_text') or response.output_text is None:
                return f"[OpenAI API error] Unexpected response format: {response}"
            if hasattr(response, 'usage'):
                print(f"OpenAI API usage: {response.usage}")
            return response.output_text.strip()
        except OpenAIError as e:
            # This is a more specific error type for OpenAI API errors:
            return f"[OpenAI API error] {str(e)}"
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

        # Dynamic prompting scheme based on the data type:
        if dataset_type == 'lidar':
            prompt_intro = ("Here are statistics and a hillshade plot derived from LiDAR elevation data. "
                        "The plot shows elevation on the left and a shaded relief view on the right.")
        else: # sentinel2
            prompt_intro = ("Here are statistics and an RGB thumbnail from a Sentinel-2 satellite median composite. "
                        "The stats include various spectral bands and a calculated NDVI (vegetation index).")
            
        # Conditionally build the content list for the user message; text input, with image_url if available:
        user_content = [
            {
                "type": "text",
                "text": (f"{prompt_intro}\n\n"
                         f"Statistics:\n{json.dumps(analysis_results.get('statistics', {}), indent=2)}\n\n"
                         "Please analyze the provided image and statistics. As an expert archaeologist and remote sensing analyst, "
                         "describe the key features, patterns, and anomalies in the landscape in plain English.")
            }
        ]

        if analysis_results.get("image"):
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{analysis_results['image']}"
                }
            })

        messages = [
            {
                "role": "system",
                "content": ("You are an expert archaeologist and remote sensing analyst. Your task is to interpret geospatial data. "
                            "You provide clear, insightful, and concise interpretations based on the data provided.")
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            print(f"Sending request to OpenRouter API for model: {model_name}...")
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions",headers=headers, json=data, timeout=120)
            print(f"Response received from OpenRouter API: {resp.status_code}")
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
def call_openai_responses(analysis_results: dict, dataset_type: str, model: str = OPENAI_DEFAULT_MODEL) -> str:
    """
    Backward-compatible: Send `analysis_results` to the OpenAI Responses API and get a plain-English description.
    Equivalent to call_model_responses(..., provider='openai').
    """
    return call_model_responses(analysis_results, dataset_type=dataset_type, provider=OPENAI_PROVIDER, model=model)
