"""Safe rotation audit/projection without changing BOP source annotations."""

import hashlib
import json

import numpy as np
import pytest

from bop_toolkit_lib.race6d_rotations import (
    audit_scene_gt, main, prepare_rotation, rotation_metrics,
)


@pytest.mark.parametrize("matrix", [np.eye(3), np.diag([1., -1., -1.])])
def test_valid_rotations_are_unchanged(matrix):
    assert rotation_metrics(matrix)["valid"]
    for mode in ("preserve", "check", "normalize"):
        result, _, changed = prepare_rotation(matrix, mode)
        assert np.array_equal(result, matrix)
        assert not changed


def test_svd_repairs_only_small_drift_and_is_idempotent():
    original = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    perturbed = original.copy()
    perturbed[0, 0] += 0.001
    perturbed[1, 1] -= 0.002
    before = perturbed.copy()
    preserved, _, changed = prepare_rotation(perturbed, "preserve")
    assert np.array_equal(preserved, perturbed) and not changed
    with pytest.raises(ValueError, match="Invalid rotation"):
        prepare_rotation(perturbed, "check")
    normalized, metrics, changed = prepare_rotation(perturbed.reshape(-1), "normalize")
    assert changed and not metrics["valid"]
    assert rotation_metrics(normalized)["valid"]
    assert np.array_equal(perturbed, before)
    second, _, changed = prepare_rotation(normalized, "normalize")
    assert np.array_equal(second, normalized) and not changed
    # The polar/SVD factor is the nearest proper rotation in Frobenius norm.
    assert np.linalg.norm(normalized - perturbed) <= np.linalg.norm(original - perturbed)
    with pytest.raises(ValueError, match="exceeds"):
        prepare_rotation(perturbed, "normalize", max_change=1e-5)


@pytest.mark.parametrize("matrix,match", [
    (np.diag([-1., 1., 1.]), "reflection"),
    (np.zeros((3, 3)), "singular"),
    (2 * np.eye(3), "exceeds"),
    ([1.] * 8, "nine"),
    (np.full((3, 3), np.nan), "non-finite"),
    (np.eye(3) * 1e200, "too large"),
])
def test_unsafe_matrices_are_not_silently_repaired(matrix, match):
    with pytest.raises(ValueError, match=match):
        prepare_rotation(matrix, "normalize")


@pytest.mark.parametrize("kwargs", [
    {"mode": "wrong"}, {"tolerance": 0}, {"tolerance": float("nan")},
    {"max_change": -1}, {"max_change": float("inf")},
])
def test_invalid_options_fail(kwargs):
    with pytest.raises(ValueError):
        prepare_rotation(np.eye(3), **kwargs)


def test_audit_reads_all_instances_and_records_hash_without_writes(tmp_path):
    path = tmp_path / "000000/scene_gt.json"
    path.parent.mkdir()
    value = {"3": [
        {"obj_id": 1, "cam_R_m2c": np.eye(3).reshape(-1).tolist()},
        {"obj_id": 2, "cam_R_m2c": (1.001 * np.eye(3)).reshape(-1).tolist()},
        {"obj_id": 3, "cam_R_m2c": [0]},
    ]}
    path.write_text(json.dumps(value))
    before = path.read_bytes()
    result = audit_scene_gt(path, max_examples=1)
    assert result["instances"] == 3 and result["invalid"] == 2
    assert len(result["examples"]) == 1
    assert result["examples"][0]["bop_im_id"] == 3
    assert result["examples"][0]["gt_id"] == 1
    assert result["sha256"] == hashlib.sha256(before).hexdigest()
    report_path = tmp_path / "audit.json"
    with pytest.raises(SystemExit) as error:
        main(["--input", str(tmp_path), "--report", str(report_path)])
    assert error.value.code == 1
    assert json.loads(report_path.read_text())["invalid"] == 2
    assert path.read_bytes() == before
    report_before = report_path.read_bytes()
    with pytest.raises(SystemExit):
        main(["--input", str(path), "--report", str(report_path)])
    assert report_path.read_bytes() == report_before


def test_audit_accepts_scene_or_file_and_rejects_empty_input(tmp_path):
    scene = tmp_path / "000001"
    scene.mkdir()
    path = scene / "scene_gt.json"
    path.write_text(json.dumps({"0": [{"cam_R_m2c": np.eye(3).reshape(-1).tolist()}]}))
    for source in (scene, path):
        assert main(["--input", str(source)])["invalid"] == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        main(["--input", str(empty)])
