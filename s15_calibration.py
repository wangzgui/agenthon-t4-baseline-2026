"""Residual bands and conservative fusion; no evaluation outcomes are read."""
from __future__ import annotations

import math
import statistics
from statistics import NormalDist
from s15_semantics import infer


def quantile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    index = (len(values) - 1) * q
    low = int(index)
    return values[low] + (values[min(low + 1, len(values)-1)] - values[low]) * (index-low)


def fusion_weight(fit, statement=False):
    if fit is None:
        # A reported-quarter transfer is weaker than an out-of-time fitted rule.
        return 0.35 if statement else 0.0
    origins = getattr(fit, 'validation_origins', 0)
    reliability = origins / (origins + 6.0)
    gain = max(0.0, fit.validation_score - fit.validation_baseline)
    # Historical rule validation is NOT validation of the House blend.
    return min(0.8, 0.25 + 0.4 * reliability + 0.15 * min(1.0, gain))


def interval(task, point, suggested=None, residuals=(), origins=0, disagreement=0.0,
             prior=None, weight=0.0):
    level = float(task.get('interval_level', 0.9))
    alpha = 1-level
    z = NormalDist().inv_cdf(1-alpha/2)
    meaning = infer(task)
    suggested = suggested or {}
    lo, hi = suggested.get('lo'), suggested.get('hi')
    valid_band = (isinstance(lo, (int,float)) and isinstance(hi, (int,float))
                  and math.isfinite(lo) and math.isfinite(hi) and lo <= hi)
    # Explicitly uncalibrated fallbacks, not pseudo-residuals or guarantees.
    floor = {'bps': 8.0, 'pct': 1.0, 'ratio': 0.15, 'probability': 0.15,
             'usd': 0.10, 'level': max(0.1, abs(point)*0.15)}.get(meaning.unit, 0.1)
    if meaning.kind == 'growth':
        floor = max(floor, 10.0)
    if 'return' in meaning.target_name.lower() or 'return' in str(task.get('prompt','')).lower():
        floor = max(floor, 3.0)
    fallback = max(floor, abs(point)*(0.10 if meaning.kind=='level' else 0.2), z*disagreement)
    if residuals:
        # Pooling across entities does not create extra independent time origins.
        n = max(1, origins)
        shrink = n/(n+8.0)
        lower = min(0.0, quantile(residuals, alpha/2))
        upper = max(0.0, quantile(residuals, 1-alpha/2))
        # These residuals belong to the statistical predictor. Account for the
        # unvalidated House correction instead of claiming fused OOF calibration.
        correction = abs(point-prior) if prior is not None else 0.0
        left = max(floor*0.25, shrink*(-lower)+(1-shrink)*fallback)
        right = max(floor*0.25, shrink*upper+(1-shrink)*fallback)
        extra = (1-weight)*fallback + min(correction, 2*fallback) + z*disagreement
        left, right = max(left, extra), max(right, extra)
        source = 'heldout_statistical_residuals_with_blend_guard'
    else:
        left = right = fallback
        if valid_band:
            center = (lo+hi)/2
            left = max(left, center-lo)
            right = max(right, hi-center)
        source = 'uncalibrated_semantic_or_house_band'
    lower, upper = point-left, point+right
    if meaning.unit == 'probability':
        lower, upper = max(0.0, lower), min(1.0, upper)
    return {'level':level, 'lo':lower, 'hi':upper}, source
