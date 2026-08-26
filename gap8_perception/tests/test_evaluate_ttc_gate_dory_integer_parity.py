import numpy as np

from gap8_perception.evaluate_ttc_gate_dory_integer_parity import parity_metrics


def test_decoded_gate_and_ttc_integer_parity_metrics():
    gate_float = np.zeros((1, 8, 20, 20), np.float32)
    gate_integer = gate_float.copy()
    # Three peaks stay fixed; BR moves one cell right in the integer graph.
    for channel, (y, x) in enumerate(((1, 2), (3, 4), (5, 6), (7, 8))):
        gate_float[0, channel, y, x] = 5.0
        gate_integer[0, channel, y, x + (channel == 2)] = 5.0
    gate_float[0, 4:8] = np.asarray((0.0, 2.0, -2.0, 0.0))[:, None, None]
    gate_integer[0, 4:8] = np.asarray((0.0, 1.0, -1.0, 0.0))[:, None, None]

    ttc_float = np.zeros((1, 7, 20, 20), np.float32)
    ttc_integer = ttc_float.copy()
    ttc_float[:, 0] = -1.0
    ttc_integer[:, 0] = 0.0
    ttc_float[:, 1] = 1.0
    ttc_integer[:, 1] = 1.0
    ttc_integer[:, 2] = 3.0
    ttc_float[:, 4:7] = np.asarray((0.0, 0.0, 2.0))[:, None, None]
    ttc_integer[:, 4:7] = np.asarray((0.0, 0.0, 1.0))[:, None, None]

    report = parity_metrics(gate_float, gate_integer, ttc_float, ttc_integer)
    assert report["gate"]["heatmap_peak_cell_exact_agreement"] == 0.75
    assert report["gate"]["heatmap_peak_cell_displacement_mean"] == 0.25
    assert np.isclose(report["gate"]["heatmap_peak_cell_displacement_p95"], 0.85)
    assert report["gate"]["visibility_probability_mae"] > 0.0
    assert report["gate"]["visibility_thresholded_agreement"] == 1.0
    assert report["ttc"]["inverse_ttc_softplus_mae"] > 0.0
    assert report["ttc"]["inverse_depth_softplus_mae"] == 0.0
    assert report["ttc"]["flow_mae"] == 1.5
    assert report["ttc"]["flow_epe_mean"] == 3.0
    assert report["ttc"]["risk_softmax_class_agreement"] == 1.0
    assert report["ttc"]["risk_critical_probability_mae"] > 0.0
