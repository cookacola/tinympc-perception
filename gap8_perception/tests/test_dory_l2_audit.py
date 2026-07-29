from gap8_perception.audit_dory_l2 import simulate_directional_allocator


def test_directional_allocator_counts_opposite_live_ends():
    arrays = {
        "activations_size": [10, 20],
        "activations_out_size": [20, 5],
        "weights_size": [3, 4],
        "layer_with_weights": [1, 1],
        "L3_input_layers": [1, 0],
        "L3_output_layers": [0, 0],
        "branch_input": [0, 0],
        "branch_output": [0, 0],
        "branch_change": [0, 0],
    }
    report = simulate_directional_allocator(arrays)
    assert report["peak_live_bytes"] == 33
    assert report["final_begin_bytes"] == 5
    assert report["final_end_bytes"] == 0
