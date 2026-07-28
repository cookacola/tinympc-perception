import cv2
import numpy as np

from gap8_perception.gate_geometry import gate_projection, rotation_error_degrees


def test_known_gate_projection_supports_metric_pnp_round_trip():
    K = np.array([[183.25, 0, 80], [0, 183.25, 80], [0, 0, 1]], np.float64)
    projection = gate_projection(
        0,
        eye=[0.2, -0.45, 0.7],
        target=[-1.3, -0.45, 0.55],
        camera_matrix=K,
        distortion=np.zeros(5),
    )
    success, rvec, tvec = cv2.solvePnP(
        projection["object_points_ordered"],
        projection["pixels_ordered"],
        K,
        np.zeros(5),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    assert success
    assert np.linalg.norm(
        tvec[:, 0] - projection["translation_camera_from_gate"]
    ) < 1e-6
    assert rotation_error_degrees(
        cv2.Rodrigues(rvec)[0], projection["rotation_camera_from_gate"]
    ) < 1e-4
