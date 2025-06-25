"""
Implements anomaly scoring and ranking for H3 grid cells.
Two modes of operations for this scoring and ranking mechanism:
 - weighted mode: scores the cells using a weighted linear combination of features.
 - iforest mode: scores the cells using an unsupervised Isolation Forest model.
"""
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest    

# ===============================================================================
# CONFIGURATION
# ===============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================================================
# FEATURE DICTIONARIES 
# ===============================================================================
# Per-cell normalized features with weights for anomaly scoring:
PERCELL_FEATURES: Dict[str, float] = {
    # Canopy structure features:
    "height_variability_norm": 0.15,
    "canopy_height_range_norm": 0.15,
    "shot_density_norm": 0.12,
    "rh98_height_difference_norm": 0.08,  # Updated from rh98_height_ratio_norm
    
    # Disturbance/deforestation features:
    "deforest_impact_norm": 0.18,  # Increased from 0.17 (most important indicator)
    
    # Terrain/elevation features (SRTM-derived):
    "terrain_complexity_norm": 0.12,  # Increased from 0.11
    "canopy_elevation_difference_norm": 0.10,  # Updated from canopy_elevation_ratio_norm
    "slope_canopy_interaction_norm": 0.08,
    "mean_slope_norm": 0.02,  # Decreased from 0.05 (least informative terrain feature)
}

# ===============================================================================
# SCORING FUNCTIONS: LINEAR COMBINATION AND ISOLATION FOREST
# ===============================================================================
def weighted_score(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    """
    Computing simple weighted linear combination scores for the features.
    Takes the feature DataFrame and the weights dictionary containing the weights 
    for each respective feature, then returns and array of the weighted scores.
    """
    score = np.zeros(len(df))
    for feat, weight in weights.items():
        if feat in df.columns:
            feature_values = df[feat].fillna(0).to_numpy()
            score += weight * feature_values
        else:
            logger.warning(f"Feature '{feat}' missing; weight ignored.")    
    return score

def iforest_score(df: pd.DataFrame, features: List[str], 
                  contamination: float = 0.05, random_state: int = 424242) -> np.ndarray:
    """
    Computation of the anomaly scores via the Isolation Forest model. Takes the 
    feature DataFrame and the list of features to use, then returns normalized 
    anomaly scores where higher values indicate more anomalous cells.
    """
    # Preparing the feature matrix, then fitting the Isolation Forest model:
    X = df[features].fillna(0)
    iforest = IsolationForest(
        n_estimators=100, 
        contamination=contamination, 
        random_state=random_state,
        n_jobs=1
    )
    iforest.fit(X)
    
    # Getting the anomaly scores (inverting so higher = more anomalous), then normalizing to [0, 1] range:
    raw_scores = -iforest.score_samples(X)
    norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    
    # Finally, returning the normalized anomaly scores:
    return norm_scores

# ===============================================================================
# MAIN SCORING AND RANKING PIPELINE
# ===============================================================================

def score_cells(percell_df: pd.DataFrame, method: str = 'weighted', **kwargs) -> pd.DataFrame:
    """Adding the computed anomaly scores to the per-cell DataFrame."""
    result_df = percell_df.copy()
    
    if method == 'weighted':
        weights = kwargs.get('weights', PERCELL_FEATURES)
        result_df['score'] = weighted_score(percell_df, weights)
        
    elif method == 'iforest':
        feature_cols = kwargs.get('feature_cols')
        if feature_cols is None:
            logger.warning("No feature columns specified; using all available features in PERCELL_FEATURES dict.")
            feature_cols = [col for col in PERCELL_FEATURES.keys() if col in percell_df.columns]
        
        result_df['score'] = iforest_score(
            percell_df, 
            feature_cols,
            kwargs.get('contamination', 0.05),
            kwargs.get('random_state', 424242)
        )
        
    else:
        raise ValueError(f"Unknown scoring method: {method}")
    
    # Finally, returning the DataFrame with the added 'score' column:
    return result_df

def rank_cells(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Takes the per-cell DataFrame (which contains the anomaly scores), as well as the 
    number of top N cells to return, and then returns the top N cells, sorted in descending
    order.
    """
    if 'score' not in df.columns:
        raise ValueError("Score column not found. Call score_cells() first.")
    return df.sort_values('score', ascending=False).head(n).reset_index(drop=True)


