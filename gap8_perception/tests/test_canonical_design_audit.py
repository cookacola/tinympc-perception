from pathlib import Path

from gap8_perception.audit_canonical_design import audit


def test_canonical_source_tree_matches_completed_design():
    report = audit(Path(__file__).parents[2])
    assert report["passed"]
    assert report["output_nchw"] == [1, 12, 15, 20]
