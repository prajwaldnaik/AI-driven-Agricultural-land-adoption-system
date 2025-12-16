from dataclasses import dataclass
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

@dataclass
class ScorerConfig:
    w_ph: float = 0.25
    w_rain: float = 0.20
    w_temp: float = 0.20
    w_water: float = 0.15
    w_shade: float = 0.10
    w_slope: float = 0.10

def clamp_score(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def range_score(val: float, lo: float, hi: float, soft: float = 0.2) -> float:
    """
    Score 1.0 inside [lo, hi]. Soft decay outside by relative distance.
    """
    if pd.isna(val) or pd.isna(lo) or pd.isna(hi):
        return 0.5
    if lo <= val <= hi:
        return 1.0
    width = max(1e-6, hi - lo)
    if val < lo:
        d = (lo - val) / width
    else:
        d = (val - hi) / width
    return clamp_score(max(0.0, 1.0 - soft * d))

def categorical_score(val: str, allowed: str) -> float:
    """
    If val is in allowed set -> 1.0; otherwise 0.5 (neutral).
    allowed is pipe-separated string (e.g., "shade|full sun").
    """
    if not allowed or not isinstance(allowed, str):
        return 0.5
    allowed_set = {s.strip().lower() for s in allowed.split('|') if s.strip()}
    v = (val or '').strip().lower()
    return 1.0 if v in allowed_set else 0.5

def compute_crop_score(row: pd.Series, inputs: Dict[str, Any], cfg: ScorerConfig) -> Tuple[float, Dict[str, float]]:
    s = {}
    s['pH'] = range_score(inputs['ph'], row['ph_min'], row['ph_max'])
    s['rain'] = range_score(inputs['rain_mm'], row['rain_min'], row['rain_max'])
    user_t = np.mean([inputs['tmin'], inputs['tmax']])
    s['temp'] = range_score(user_t, row['tmin'], row['tmax'])
    s['water'] = categorical_score(inputs['water'], row['water_need'])

    shade_user = 'shade' if inputs['shade_percent'] >= 30 else 'full sun'
    s['shade'] = categorical_score(shade_user, row['shade_pref'])

    slope_cat = 'gentle' if inputs['slope_percent'] <= 5 else ('moderate' if inputs['slope_percent'] <= 10 else 'steep')
    s['slope'] = categorical_score(slope_cat, row['slope_ok'])

    total = (cfg.w_ph*s['pH'] + cfg.w_rain*s['rain'] + cfg.w_temp*s['temp'] +
             cfg.w_water*s['water'] + cfg.w_shade*s['shade'] + cfg.w_slope*s['slope'])
    return float(total), s

def recommend(df: pd.DataFrame, inputs: Dict[str, Any], top_k: int = 8) -> pd.DataFrame:
    cfg = ScorerConfig()
    rows = []
    for _, row in df.iterrows():
        score, parts = compute_crop_score(row, inputs, cfg)
        rows.append({
            'crop': row['crop'],
            'score': round(score, 3),
            'reason': f"pH:{parts['pH']:.2f} rain:{parts['rain']:.2f} temp:{parts['temp']:.2f} "
                      f"water:{parts['water']:.2f} shade:{parts['shade']:.2f} slope:{parts['slope']:.2f}",
            'spacing': row.get('spacing', ''),
            'intercrops': row.get('intercrops', ''),
            'notes': row.get('notes', '')
        })
    out = pd.DataFrame(rows).sort_values('score', ascending=False).head(top_k).reset_index(drop=True)
    return out
