"""Training asset preparation preserves original BOP geometry and annotations."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bop_toolkit_lib.race6d_models import (
    P_OBJECTS,
    find_alignment,
    install_bundle,
    prepare_models,
    shift_model_info,
    shift_ply,
    symmetry_axis,
)
from bop_toolkit_lib.race6d_keypoints import (
    _notebook_symmetry_axis,
    generate_keypoints,
    symmetry_axis_frame,
    validate_keypoints,
)


def _info(p=(4, -3, 2)):
    return {
        "diameter": 40,
        "min_x": -6,
        "min_y": -8,
        "min_z": -10,
        "size_x": 12,
        "size_y": 16,
        "size_z": 20,
        "symmetries_continuous": [{"axis": [0, 0, 1], "offset": list(p)}],
    }


def _write_models(path):
    plyfile = pytest.importorskip("plyfile")
    path.mkdir()
    # Point-only box: no renderer/sampling backend is needed for these tests.
    points = np.array(
        [(x, y, z) for z in (-10, 10) for y in (-8, 8) for x in (-6, 6)],
        dtype=np.float64,
    )
    vertex = np.empty(
        len(points), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("red", "u1")]
    )
    for i, axis in enumerate("xyz"):
        vertex[axis] = points[:, i]
    vertex["red"] = np.arange(len(points))
    plyfile.PlyData([plyfile.PlyElement.describe(vertex, "vertex")], text=True).write(
        str(path / "obj_000001.ply")
    )
    (path / "models_info.json").write_text(json.dumps({"1": _info()}))
    return points


def test_continuous_axis_and_discrete_point():
    point, axis, residual = symmetry_axis(_info())
    assert residual < 1e-12
    np.testing.assert_allclose(point[:2], [4, -3])
    assert abs(axis[2]) == 1
    p = np.array([4.0, -3.0, 2.0])
    transforms = []
    for rotation in (np.diag([-1.0, -1.0, 1.0]), np.diag([1.0, -1.0, -1.0])):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = rotation, (np.eye(3) - rotation) @ p
        transforms.append(m.reshape(-1).tolist())
    point, axis, residual = symmetry_axis({"symmetries_discrete": transforms})
    assert axis is None and residual < 1e-12
    np.testing.assert_allclose(point, p)


def test_all_continuous_axes_checked():
    info = _info()
    info["symmetries_continuous"].append({"axis": [0, 0, 1], "offset": [14, -3, 2]})
    assert symmetry_axis(info)[2] > 1


def test_notebook_keypoint_frame_prioritizes_first_continuous_axis():
    info = _info(p=(0, 0, 0))
    flip = np.eye(4)
    flip[:3, :3] = np.diag([1.0, -1.0, -1.0])
    mixed = dict(info, symmetries_discrete=[flip.reshape(-1).tolist()])
    # P preparation deliberately solves both types. Keypoint selection must
    # retain the notebook's axis, not redefine it as a multi-axis fixed point.
    assert symmetry_axis(mixed)[1] is None
    np.testing.assert_array_equal(_notebook_symmetry_axis(mixed)[1], [0, 0, 1])
    points = np.array(
        [(x, y, z) for z in (-10, 10) for y in (-8, 8) for x in (-6, 6)],
        dtype=np.float32,
    )
    frame = symmetry_axis_frame(mixed, points)
    assert not frame[3] and len(frame[2]) == 3
    assert all(np.isclose(m[2, 2], 1) for m in frame[2])
    a = generate_keypoints(points, info, {"type": "z_zero"})
    b = generate_keypoints(points, mixed, {"type": "z_zero"})
    np.testing.assert_array_equal(a["keypoints"], b["keypoints"])
    assert a["edges_mask"] == b["edges_mask"]


def test_notebook_point_dtype_and_discrete_fitted_center():
    from bop_toolkit_lib.race6d_keypoints import _process_points

    points = np.array(
        [(x, y, z) for z in (-9.7, 10.2) for y in (-8.3, 8.1) for x in (-6.9, 6.2)],
        dtype=np.float32,
    )
    config = {"type": "z_zero"}
    a = _process_points(points, config)
    b = generate_keypoints(points, {}, config)
    np.testing.assert_array_equal(a["keypoints"], b["keypoints"])
    assert b["ply_points"].dtype == np.float32
    transforms = []
    for rotation, offset in [
        (np.diag([-1.0, -1.0, 1.0]), [0.1, 0, 0]),
        (np.diag([1.0, -1.0, -1.0]), [0, -0.1, 0]),
    ]:
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, offset
        transforms.append(matrix.reshape(-1).tolist())
    info = {"symmetries_discrete": transforms}
    point, axis, residual = _notebook_symmetry_axis(info)
    assert axis is None and residual > 0.01
    np.testing.assert_allclose(point, [0.05, -0.025, 0], atol=1e-12)


def test_approximate_symmetry_audit_does_not_block_unchanged_origin(tmp_path):
    source = tmp_path / "models"
    _write_models(source)
    info = _info()
    info.pop("symmetries_continuous")
    info["symmetries_discrete"] = []
    for rotation, offset in [
        (np.diag([-1.0, -1.0, 1.0]), [0.1, 0, 0]),
        (np.diag([1.0, -1.0, -1.0]), [0, -0.1, 0]),
    ]:
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, offset
        info["symmetries_discrete"].append(matrix.reshape(-1).tolist())
    (source / "models_info.json").write_text(json.dumps({"1": info}))
    # Not a P object: the approximate symmetry does not stop P=0.
    offsets, audit = find_alignment(source, [1])
    assert offsets == {"1": [0.0, 0.0, 0.0]}
    assert not audit["1"]["alignment_applied"] and not audit["1"]["p_target"]
    # The same metadata must still stop an actual origin shift.
    with pytest.raises(ValueError, match="inconsistent"):
        find_alignment(source, [1], p_object_ids=[1])
    with pytest.raises(ValueError, match="inconsistent"):
        find_alignment(source, [1], p_values={"1": [0.05, -0.025, 0]})


def test_identity_not_an_axis_and_invalid_symmetry():
    assert (
        symmetry_axis({"symmetries_discrete": [np.eye(4).reshape(-1).tolist()]}) is None
    )
    with pytest.raises(ValueError, match="continuous"):
        symmetry_axis(
            {"symmetries_continuous": [{"axis": [0, 0, 0], "offset": [0, 0, 0]}]}
        )
    reflection = np.eye(4)
    reflection[0, 0] = -1
    with pytest.raises(ValueError, match="discrete"):
        symmetry_axis({"symmetries_discrete": [reflection.reshape(-1).tolist()]})


def test_shifted_metadata_and_projection_identity():
    info = _info()
    p = np.array([4.0, -3.0, 2.0])
    rotation = np.diag([-1.0, -1.0, 1.0])
    symmetry = np.eye(4)
    symmetry[:3, :3], symmetry[:3, 3] = rotation, (np.eye(3) - rotation) @ p
    info["symmetries_discrete"] = [symmetry.reshape(-1).tolist()]
    shifted = shift_model_info(info, p)
    assert info["min_x"] == -6
    assert shifted["min_x"] == -10
    np.testing.assert_allclose(shifted["symmetries_continuous"][0]["offset"], 0)
    np.testing.assert_allclose(
        np.array(shifted["symmetries_discrete"][0]).reshape(4, 4)[:3, 3], 0
    )
    vertices = np.random.default_rng(5).normal(size=(20, 3))
    t = np.array([20.0, 30.0, 600.0])
    np.testing.assert_allclose(
        (vertices - p) @ rotation.T + (t + rotation @ p),
        vertices @ rotation.T + t,
        atol=1e-12,
    )
    np.testing.assert_allclose((t + rotation @ p) - rotation @ p, t)


def test_ply_preserves_source_properties_and_zero_shift(tmp_path):
    plyfile = pytest.importorskip("plyfile")
    path = tmp_path / "source"
    points = _write_models(path)
    source = path / "obj_000001.ply"
    before = source.read_bytes()
    target = tmp_path / "shifted.ply"
    shift_ply(source, target, [4, -3, 2])
    shifted = plyfile.PlyData.read(str(target))["vertex"].data
    np.testing.assert_allclose(
        np.column_stack([shifted[a] for a in "xyz"]), points - [4, -3, 2]
    )
    np.testing.assert_array_equal(shifted["red"], np.arange(8))
    assert source.read_bytes() == before
    shift_ply(source, tmp_path / "zero.ply", [0, 0, 0])
    assert (tmp_path / "zero.ply").read_bytes() == before


def test_keypoints_strict_geometry():
    points = np.array([(x, y, z) for z in (-10, 10) for y in (-8, 8) for x in (-6, 6)])
    result = generate_keypoints(points, {}, {"type": "z_zero"})
    edges = [result["edges_mask"][name] for name in sorted(result["edges_mask"])]
    assert validate_keypoints(result["keypoints"], edges) < 1e-10
    shifted = points + [0, 0, 100]
    with pytest.raises(ValueError, match="duplicate|twelve"):
        generate_keypoints(shifted, {}, {"type": "z_zero"})
    with pytest.raises(ValueError, match="Split"):
        generate_keypoints(points, {}, {"type": "unsupported"})
    duplicate_edges = np.asarray(edges).copy()
    duplicate_edges[1] = duplicate_edges[0]
    with pytest.raises(ValueError, match="cover all|different"):
        validate_keypoints(result["keypoints"], duplicate_edges)
    with pytest.raises(ValueError, match="override"):
        generate_keypoints(points, {}, {"type": "z_zero"}, {"upper_edges": [[-1, 0]]})


def test_bundle_complete_original_unchanged_and_refuses_repeat(tmp_path):
    pytest.importorskip("open3d")
    source = tmp_path / "original"
    _write_models(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "training"
    summary = prepare_models(source, output, "synthetic", p_values={"1": [4, -3, 2]})
    assert summary["object_ids"] == [1]
    assert not summary["official_bop_data_modified"]
    cache = json.loads((output / "cached_keypoints_3d_edge.json").read_text())
    assert len(cache["keypoints_3d"]["1"]) == 32 and len(cache["edges_mask"]["1"]) == 12
    contract = json.loads((output / "model_contract.json").read_text())
    assert (
        contract["paligned_values"]
        == contract["aligned_offsets"]
        == {"1": [4.0, -3.0, 2.0]}
    )
    assert contract["need_aligned"]
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    # P only: no shifted mesh/models_info is published, nor the scratch folder.
    assert sorted(p.name for p in output.iterdir()) == [
        "cached_keypoints_3d_edge.json",
        "model_contract.json",
        "model_preparation.json",
        "p_values.json",
    ]
    assert not summary["meshes_written"]
    assert cache["metadata"]["coordinate_system"] == "original_bop_model_frame"
    assert cache["metadata"]["p_values"] == {"1": [4.0, -3.0, 2.0]}
    with pytest.raises(FileExistsError):
        prepare_models(source, output, "synthetic")
    for prepared in (output, output / "models"):
        with pytest.raises(ValueError, match="Already prepared"):
            prepare_models(prepared, tmp_path / "double", "synthetic")


def test_install_to_dataset_folder(tmp_path):
    pytest.importorskip("open3d")
    source = tmp_path / "original"
    _write_models(source)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "cached_keypoints_3d_edge.json").write_text("{}")
    summary = prepare_models(
        source, tmp_path / "b1", "synthetic", p_values={"1": [4, -3, 2]},
        install_to=dataset,
    )
    installed = summary["installed"]
    assert installed["cached_keypoints_3d_edge.json"]["action"] == "installed"
    assert installed["p_values.json"]["action"] == "installed"
    assert "backup" not in installed["p_values.json"]
    backup = Path(installed["cached_keypoints_3d_edge.json"]["backup"])
    assert backup.read_text() == "{}"
    for name in ("cached_keypoints_3d_edge.json", "p_values.json"):
        assert (dataset / name).read_bytes() == (tmp_path / "b1" / name).read_bytes()
    assert json.loads((dataset / "p_values.json").read_text()) == {"1": [4.0, -3.0, 2.0]}
    again = install_bundle(tmp_path / "b1", dataset)
    assert {v["action"] for v in again.values()} == {"unchanged"}
    assert sorted(p.name for p in dataset.iterdir()) == sorted(
        ["cached_keypoints_3d_edge.json", "p_values.json", backup.name]
    )
    with pytest.raises(FileNotFoundError):
        install_bundle(tmp_path / "b1", tmp_path / "missing")


def test_p_only_for_target_objects(tmp_path):
    pytest.importorskip("open3d")
    source = tmp_path / "original"
    _write_models(source)
    target, audit = find_alignment(source, [1], p_object_ids=[1])
    assert np.any(target["1"]) and audit["1"]["p_target"] and audit["1"]["alignment_applied"]
    kept, audit = find_alignment(source, [1])
    assert kept["1"] == [0.0, 0.0, 0.0] and not audit["1"]["p_target"]
    explicit, _ = find_alignment(source, [1], p_values={"1": [4, -3, 2]}, p_object_ids=[1])
    assert explicit["1"] == [4.0, -3.0, 2.0]
    with pytest.raises(ValueError, match="not selected"):
        find_alignment(source, [1], p_object_ids=[2])
    assert P_OBJECTS == {"ycbv": (19, 20), "itodd": (2,)}


def test_failure_publishes_nothing(tmp_path):
    pytest.importorskip("open3d")
    source = tmp_path / "original"
    _write_models(source)
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="obj 1"):
        prepare_models(source, output, "synthetic", p_values={"1": [0, 0, 100]})
    assert not output.exists()
    assert not list(tmp_path.glob(".bad.*"))


def test_alignment_residual_and_explicit_no_shift(tmp_path):
    source = tmp_path / "models"
    _write_models(source)
    offsets, audit = find_alignment(source, [1], p_object_ids=[1], p_values={})
    assert offsets == {"1": [0.0, 0.0, 0.0]}
    assert not audit["1"]["alignment_applied"]
    info = _info()
    info["symmetries_continuous"].append({"axis": [0, 0, 1], "offset": [14, -3, 2]})
    (source / "models_info.json").write_text(json.dumps({"1": info}))
    with pytest.raises(ValueError, match="inconsistent"):
        find_alignment(source, [1], p_object_ids=[1])


def test_cli_help_does_not_import_optional_geometry_backends():
    script = Path(__file__).resolve().parents[2] / "scripts/prepare_race6d_models.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "--p_values" in result.stdout


def test_keypoints_original_frame_equal_shifted_generation_plus_p(tmp_path):
    pytest.importorskip("open3d")
    from bop_toolkit_lib.race6d_keypoints import generate_cache

    source = tmp_path / "original"
    _write_models(source)
    p = np.array([4.0, -3.0, 2.0])
    output = tmp_path / "bundle"
    prepare_models(source, output, "synthetic", p_values={"1": p.tolist()})
    cache = json.loads((output / "cached_keypoints_3d_edge.json").read_text())

    # Reference: the previous behavior, generating on a v-P mesh/models_info copy.
    shifted = tmp_path / "shifted"
    shifted.mkdir()
    shift_ply(source / "obj_000001.ply", shifted / "obj_000001.ply", p)
    info = json.loads((source / "models_info.json").read_text())
    (shifted / "models_info.json").write_text(
        json.dumps({"1": shift_model_info(info["1"], p)})
    )
    reference = generate_cache(shifted, [1], "synthetic")
    np.testing.assert_allclose(
        np.asarray(cache["keypoints_3d"]["1"]) - p,
        reference["keypoints_3d"]["1"],
        atol=1e-9,
    )
    assert cache["edges_mask"] == reference["edges_mask"]

    # Zero P leaves the cache identical to generation on the original mesh.
    zero = tmp_path / "zero"
    prepare_models(source, zero, "synthetic", p_values={})
    direct = generate_cache(source, [1], "synthetic")
    zero_cache = json.loads((zero / "cached_keypoints_3d_edge.json").read_text())
    assert zero_cache["keypoints_3d"] == direct["keypoints_3d"]
