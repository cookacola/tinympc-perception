"""Controller-side speed conditioning for the single-input DORY network."""

from __future__ import annotations

import numpy as np


def collision_probability_from_range(
    inverse_range: np.ndarray,
    base_hazard_probability: np.ndarray,
    uncertainty: np.ndarray,
    body_speed_mps: float,
    horizon_s: float,
    latency_s: float,
    max_range_m: float = 6.0,
    transition_m: float = 0.15,
    nominal_target_speed_mps: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return speed-conditioned danger probability and estimated TTC.

    The DORY CNN predicts image-only inverse range. The controller then uses
    the current state and planning timing to decide whether that range is
    reachable within the TinyMPC horizon.
    """
    inv = np.clip(np.asarray(inverse_range, np.float32), 0.0, 1.0)
    base = np.clip(np.asarray(base_hazard_probability, np.float32), 0.0, 1.0)
    sigma = np.clip(np.asarray(uncertainty, np.float32), 0.0, 1.0)
    range_m = (1.0 - inv) * max_range_m
    speed = max(float(body_speed_mps), 1e-3)
    # Range belongs to the captured frame. Motion during perception/control
    # latency therefore increases, rather than decreases, the distance that
    # may be traversed before the end of the next MPC horizon.
    reachable_m = speed * max(float(horizon_s) + float(latency_s), 0.0)
    nominal_reachable_m = max(float(nominal_target_speed_mps), 1e-3) * max(
        float(horizon_s) + 0.08, 0.0
    )
    softened_margin = transition_m * (1.0 + 2.0 * sigma)
    geometric = 1.0 / (
        1.0 + np.exp(np.clip((range_m - reachable_m) / softened_margin, -30, 30))
    )
    base_logit = np.log(np.clip(base, 1e-5, 1 - 1e-5) / np.clip(1 - base, 1e-5, 1))
    adjusted = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                -(base_logit + (reachable_m - nominal_reachable_m) / softened_margin),
                -30,
                30,
            )
        )
    )
    # The nominal rollout target captures acceleration and current-direction
    # effects; the range term conservatively recovers hazards that only become
    # reachable above the nominal training speed.
    probability = np.maximum(adjusted, geometric)
    effective_range = np.maximum(0.0, range_m - speed * float(latency_s))
    ttc_s = effective_range / speed
    return probability.astype(np.float32), ttc_s.astype(np.float32)
