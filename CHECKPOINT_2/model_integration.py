"""
Model integration for comprehensive anomalous archaeological feature detection.

Centralized prompting utilities that sends the top-N anomaly H3 cells, along with regional LiDAR and 
Sentinel-2 context to either OpenAI or OpenRouter models via its APIs. The model returns a structured and 
comprehensive assessment indicating whether each cell is likely to contain any anomalous archaeological
features, all in a JSON format.
"""
from __future__ import annotations
import os
import json
import hashlib
import logging
from pathlib import Path
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ===============================================================================
# OpenAI/ OpenRouter ENVIRONMENT SETUP AND LOGGING CONFIGURATION
# ===============================================================================
load_dotenv()
OPENAI_PROVIDER = "openai"
OPENROUTER_PROVIDER = "openrouter"
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "google/gemma-3-27b-it")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Logging configuration; path to save a JSONL log of the LLM prompts and responses.
LOG_PATH = Path(os.getenv("LLM_LOG_PATH", "llm_prompt_log.jsonl"))
# Only create directory if LOG_PATH has a parent directory
if LOG_PATH.parent != Path('.'):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================================================
# LLM PROMPT PIPELINE: SYSTEM PROMPT, PER-CELL INSTRUCTIONS, MESSAGE BUILDER
# ===============================================================================
# Main system prompt for the provided LLM:
SYSTEM_PROMPT = (
    "You are a senior Amazonian archaeologist and remote-sensing scientist. "
    "Using canopy-structure (GEDI), terrain (SRTM), disturbance (PRODES) metrics "
    "plus regional LiDAR elevation and Sentinel-2 spectral composites, decide "
    "whether each H3 grid cell encloses a likely anthropogenic feature such as "
    "geoglyphs, ADE soil patches, earthworks, mounded villages, or engineered "
    "drainage.  Be explicit about which variables drive your reasoning."  
)

# Per-cell specific instructions for the provided LLM:
CELL_INSTRUCTIONS = (
    "For the cell JSON below:\n"
    "1. Rate *Archaeological Potential* as **high / medium / low**.\n"
    "2. Give a bullet-point *Rationale* referencing at least three metrics "
    "(e.g., canopy_height_range_norm, terrain_complexity_norm, deforest_impact_norm).\n"
    "3. Suggest *Field Verification Priority* on a scale **1-5** (5 = inspect first).\n"
    "Return a JSON object with keys `potential`, `rationale`, `priority`."
)

# Message builder for the provided LLM:
def build_messages(cell: Dict[str, Any], lidar_s2_ctx: Dict[str, Any], regional_assessment: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Function that facilitates the construction of a multimodal message list, suitable for both OpenAI and
    OpenRouter models. 
    """
    # Cell text; per-cell feature vector from GEDI, STRM, PRODES + per-cell instructions, in JSON format:
    cell_text  =  "### Anomaly Cell Feature Vector\n" \
        + json.dumps(cell, indent=2) \
        + f"\n\n{CELL_INSTRUCTIONS}"
    user_content: List[Dict[str, Any]] = []

    # If a regional assessment is provided, prepend it to the user content
    if regional_assessment is not None:
        user_content.append({"type": "text", "text": f"### Regional Assessment (LLM-generated):\n{regional_assessment}\n"})

    user_content.append({"type": "text", "text": cell_text})

    # Images from regional LiDAR and Sentinel-2 composites:
    for key in ("image", "ndvi_image", "false_color_image"):
        b64_img = lidar_s2_ctx.get(key)
        if b64_img:
            user_content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

    # Statistics and metadata from regionalLiDAR and Sentinel-2 composites, in JSON format:
    stats_text = "\n\n### Regional LiDAR and Sentinel-2 Statistics\n" \
        + json.dumps(lidar_s2_ctx.get("statistics", {}), indent=2)
    user_content.append({"type": "text", "text": stats_text})

    # Finally, returning the complete message for the LLM; system + user content:
    return [{"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "user", "content": user_content}
           ]

# ===============================================================================
# OPENAI API MODEL CALL
# ===============================================================================
def openai_model_call(
    messages: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 128000,   
    **kwargs
) -> str:
    """
    Sends the provided message(s) to a specified OpenAI model via its API, and returns the LLM's response.
    """
    if openai_client is None:
        raise ValueError("OpenAI client is not initialized and/or API key is missing. Cannot call OpenAI API.")
    try:
        logger.info(f"Sending request to OpenAI Responses API for model: {model_name}...")
        llm_response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        logger.info("Response received from OpenAI Chat Completions API.")
        return llm_response.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"[OpenAI API error] {str(e)}"   

# ===============================================================================
# OPENROUTER API MODEL CALL
# ===============================================================================
def openrouter_model_call(
    messages: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 128000,   
    **kwargs
) -> str:
    """
    Sends the provided message(s) to a specified OpenRouter model via its API, and returns the LLM's response.
    """
    if not OPENROUTER_API_KEY:
        return "[OpenRouter API error] OpenRouter API key is missing."
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }    
    
    try:
        logger.info(f"Sending request to OpenRouter API for model: {model_name}...")
        llm_response = requests.post("https://openrouter.ai/api/v1/chat/completions",headers=headers, json=payload, timeout=120)
        logger.info(f"Response received from OpenRouter API: {llm_response.status_code}")
        llm_response_json = llm_response.json()
        if llm_response.status_code != 200:
            return f"[OpenRouter API error] {llm_response_json.get('error', llm_response.text)}"
        return llm_response_json["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[OpenRouter API error] {str(e)}"

# ===============================================================================
# UNIFIED WRAPPER FOR API MODEL CALLS
# ===============================================================================
def llm_model_call(
    messages: List[Dict[str, Any]],
    provider: str = OPENAI_PROVIDER,    
    model_name: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 128000,   
    **kwargs,
) -> str:
    """
    Convenient unified wrapper for the API model calls. Sends the already-build chat messages to the 
    chosen LLM provider, and returns the LLM's response.
    """
    if provider == OPENAI_PROVIDER:
        model_name = model_name or OPENAI_DEFAULT_MODEL
        return openai_model_call(messages, model_name, temperature, max_tokens, **kwargs)
    elif provider == OPENROUTER_PROVIDER:
        model_name = model_name or OPENROUTER_DEFAULT_MODEL
        return openrouter_model_call(messages, model_name, temperature, max_tokens, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ===============================================================================
# MAIN FUNCTION: FROM TOP-N SCORING CELLS TO LLM RESPONSE
# ===============================================================================
def analyze_top_n_cells(
    top_n_cells: List[Dict[str, Any]],
    lidar_s2_ctx: Dict[str, Any],
    provider: str = OPENROUTER_PROVIDER,     # Lets default to OpenRouter API for now.
    model_name: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 128000,   # Default for now, change depending on specific model's ctx window.
    save_log: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Faciliates the full process of anomalous archaeological feature detection via LLM usage.
    Loops through the top-N best scoring cells, builds a comprehensive multimodal message to send to a 
    specified LLM provider for inquiry, and collects the structured LLM assessment(s) for logging.
    Each resultant logging entry is a dictionary with the following keys:
        - cell_id: the cell feature vector's ID.
        - provider: the specific LLM provider used.
        - model_name: the specific LLM model used from the provider.
        - message_hash: the hash of the message used to generate the assessment.
        - llm_response: the LLM's response to the given message.
    """
    results: List[Dict[str, Any]] = []

    for idx, cell in enumerate(top_n_cells, start=1):
        cell_id = cell.get("h3_cell", f"cell_{idx}")
        logger.info(f"Sending cell {cell_id} and regional context to {provider} API...")
        
        messages = build_messages(cell, lidar_s2_ctx)
        llm_response = llm_model_call(messages, provider, model_name, temperature, max_tokens, **kwargs)
        
        # Log Entry:
        entry = {
            "cell_id": cell_id,
            "provider": provider,
            "model_name": model_name or (OPENAI_DEFAULT_MODEL if provider == OPENAI_PROVIDER 
                                        else OPENROUTER_DEFAULT_MODEL),
            "message_hash": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
            "llm_response": llm_response,
        }
        results.append(entry)

    # Save logged entries to specified log file path:
    if save_log:
        with LOG_PATH.open("a") as entry_path:
            for entry in results:
                entry_path.write(json.dumps(entry) + "\n")
        logger.info(f"Logged {len(results)} entries (message-response pairs) to {LOG_PATH}")
    return results

# ===============================================================================
# EFFICIENT BATCH ANALYSIS: SEND REGIONAL CONTEXT ONCE FOR ALL CELLS
# ===============================================================================
def build_regional_context_message(lidar_s2_ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a message for the LLM to assess only the regional LiDAR and Sentinel-2 context.
    """
    prompt = (
        "You are a senior Amazonian archaeologist and remote-sensing scientist. "
        "Given ONLY the following regional LiDAR and Sentinel-2 statistics, images, and metadata, "
        "summarize the regional context and highlight any features, anomalies, or patterns that could "
        "influence archaeological potential in the area. Return a concise summary that can be used as context for further cell-level analysis."
    )
    user_content: List[Dict[str, Any]] = []
    # Images from regional LiDAR and Sentinel-2 composites:
    for key in ("image", "ndvi_image", "false_color_image"):
        b64_img = lidar_s2_ctx.get(key)
        if b64_img:
            user_content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
    # Statistics and metadata from regionalLiDAR and Sentinel-2 composites, in JSON format:
    stats_text = "\n\n### Regional LiDAR and Sentinel-2 Statistics\n" \
        + json.dumps(lidar_s2_ctx.get("statistics", {}), indent=2)
    user_content.append({"type": "text", "text": stats_text})
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content}
    ]

def analyze_top_n_cells_batch(
    top_n_cells: List[Dict[str, Any]],
    lidar_s2_ctx: Dict[str, Any],
    provider: str = OPENROUTER_PROVIDER,
    model_name: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 128000,
    save_log: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    New batch analysis: first, LLM assesses regional LiDAR/Sentinel-2 context; then, each cell is assessed using this regional summary as context.
    Returns a dict with the regional assessment and all cell assessments, plus logging metadata.
    """
    logger.info(f"Step 1: Sending regional context to {provider} API...")
    regional_messages = build_regional_context_message(lidar_s2_ctx)
    regional_assessment = llm_model_call(regional_messages, provider, model_name, temperature, max_tokens, **kwargs)

    logger.info("Step 2: Assessing each cell with regional context...")
    cell_results = []
    for idx, cell in enumerate(top_n_cells):
        cell_id = cell.get("h3_cell", f"cell_{idx+1}")
        messages = build_messages(cell, lidar_s2_ctx, regional_assessment=regional_assessment)
        llm_response = llm_model_call(messages, provider, model_name, temperature, max_tokens, **kwargs)
        entry = {
            "cell_id": cell_id,
            "llm_response": llm_response,
            "messages": messages
        }
        cell_results.append(entry)

    batch_entry = {
        "batch_id": f"batch_{len(top_n_cells)}_cells",
        "cell_ids": [cell.get("h3_cell", f"cell_{idx+1}") for idx, cell in enumerate(top_n_cells)],
        "provider": provider,
        "model_name": model_name or (OPENAI_DEFAULT_MODEL if provider == OPENAI_PROVIDER else OPENROUTER_DEFAULT_MODEL),
        "regional_assessment": regional_assessment,
        "cell_assessments": cell_results,
        "num_cells": len(top_n_cells)
    }

    # Save batch entry to log
    if save_log:
        with LOG_PATH.open("a") as entry_path:
            entry_path.write(json.dumps(batch_entry) + "\n")
        logger.info(f"Logged batch analysis of {len(top_n_cells)} cells (regional + per-cell) to {LOG_PATH}")

    return batch_entry

