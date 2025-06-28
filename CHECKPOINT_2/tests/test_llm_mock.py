"""Offline tests for LLM message construction.

These tests avoid API calls and live geospatial services. They verify that the
current model_integration helpers build the expected chat-message structure.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pytest.importorskip("openai", reason="model_integration imports OpenAI client")

from model_integration import build_messages, build_regional_context_message


def create_mock_lidar_s2_context():
    return {
        "statistics": {
            "lidar_stats": {
                "mean_elevation": 145.2,
                "std_elevation": 12.8,
                "min_elevation": 120.5,
                "max_elevation": 180.3,
            },
            "sentinel2_stats": {
                "ndvi_stats": {"mean": 0.76, "std": 0.12, "min": 0.45, "max": 0.95},
                "cloud_coverage": 5.2,
            },
            "regional_summary": {
                "total_area_km2": 25.6,
                "forest_coverage_pct": 78.4,
                "deforestation_pct": 3.2,
            },
        },
        "image": None,
        "ndvi_image": None,
        "false_color_image": None,
    }


def create_mock_cell():
    return {
        "h3_cell": "898aa919897ffff",
        "lat": -5.15,
        "lon": -59.85,
        "shot_count": 12,
        "mean_canopy_height": 24.3,
        "height_variability_norm": 0.62,
        "terrain_complexity_norm": 0.48,
        "deforest_impact_norm": 0.71,
        "score": 0.68,
    }


def test_individual_message_structure():
    messages = build_messages(create_mock_cell(), create_mock_lidar_s2_context())

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert isinstance(messages[1]["content"], list)
    assert any(item["type"] == "text" for item in messages[1]["content"])

    user_text = "\n".join(
        item["text"] for item in messages[1]["content"] if item["type"] == "text"
    )
    assert "898aa919897ffff" in user_text
    assert "mean_canopy_height" in user_text


def test_regional_context_message_structure():
    messages = build_regional_context_message(create_mock_lidar_s2_context())

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert isinstance(messages[1]["content"], list)
    assert any(item["type"] == "text" for item in messages[1]["content"])

    user_text = "\n".join(
        item["text"] for item in messages[1]["content"] if item["type"] == "text"
    )
    assert "LiDAR" in user_text or "lidar" in user_text
