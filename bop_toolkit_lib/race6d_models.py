"""Determine P and generate original-frame keypoints from original BOP models.

Ported from RACE6D/sub/compute_symmetry_centers.py and process_models.py.
Meshes and GT stay in the original BOP frame: the bundle holds P and a keypoint
cache in that same frame, and the model loader applies P (v-P, kp-P, t+R@P).
All lengths, including P and symmetry translations, are in BOP millimeters.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def symmetry_axis(info):
    """Return a common fixed point, optional axis, and equation residual.

    Solve every discrete/continuous constraint, not just the first continuous
    axis. A translation can remove symmetry translations only if the combined
    equations have a common solution. Callers must check the residual.
    """
    equations, values = [], []
    for symmetry in info.get("symmetries_continuous", []):
        axis = np.asarray(symmetry["axis"], dtype=np.float64)
        offset = np.asarray(symmetry["offset"], dtype=np.float64)
        if (
            axis.shape != (3,)
            or offset.shape != (3,)
            or not np.isfinite(axis).all()
            or not np.isfinite(offset).all()
            or np.linalg.norm(axis) < 1e-12
        ):
            raise ValueError("Invalid continuous symmetry axis/offset")
        axis = axis / np.linalg.norm(axis)
        equation = np.eye(3) - np.outer(axis, axis)
        equations.append(equation)
        values.append(equation @ offset)
    for symmetry in info.get("symmetries_discrete", []):
        matrix = np.asarray(symmetry, dtype=np.float64).reshape(4, 4)
        rotation = matrix[:3, :3]
        if (
            not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)
        ):
            raise ValueError("Invalid discrete symmetry transform")
        equations.append(np.eye(3) - rotation)
        values.append(matrix[:3, 3])
    if not equations:
        return None
    equation = np.concatenate(equations)
    value = np.concatenate(values)
    _, singular, right = np.linalg.svd(equation, full_matrices=False)
    if singular[0] < 1e-12:
        if np.max(np.abs(value)) > 1e-9:
            raise ValueError(
                "A pure translation cannot be removed by moving the origin"
            )
        return None
    point = np.linalg.lstsq(equation, value, rcond=None)[0]
    rank = int(np.sum(singular > 1e-6 * singular[0]))
    axis = right[-1] if rank == 2 else None
    if rank < 2:
        raise ValueError("Symmetry constraints do not determine an axis or fixed point")
    return point, axis, float(np.abs(equation @ point - value).max())


def surface_centroid(path, samples=20000, seed=0):
    import trimesh

    mesh = trimesh.load(str(path), process=False)
    vertices = np.asarray(mesh.vertices)
    if not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Invalid/empty mesh: {}".format(path))
    if hasattr(mesh, "faces") and len(mesh.faces):
        points, _ = trimesh.sample.sample_surface(mesh, samples, seed=seed)
        return points.mean(axis=0)
    return vertices.mean(axis=0)


# Objects trained in a model frame moved to their symmetry axis. Only these get
# a P by default; every other object keeps P=0. T-LESS 25/26 also have an
# off-origin axis, but only their body is symmetric (a 14 mm protrusion has no
# counterpart) and RACE6D trains them as asymmetric, so they are not listed.
P_OBJECTS = {"ycbv": (19, 20), "itodd": (2,)}


def find_alignment(
    models_path, object_ids, p_object_ids=(), max_residual_mm=0.01, p_values=None
):
    """P per object: the symmetry-axis point for p_object_ids, 0 for the rest.

    p_values (explicit {obj_id: [x, y, z]} mm, omitted objects 0) replaces the
    computation.
    """
    models_path = Path(models_path)
    info = json.loads((models_path / "models_info.json").read_text())
    if not np.isfinite(max_residual_mm) or max_residual_mm <= 0:
        raise ValueError("max_residual_mm must be finite and positive")
    if p_values is not None and not isinstance(p_values, dict):
        raise ValueError("P values must be an object-ID mapping")
    explicit = (
        None if p_values is None else {str(int(k)): v for k, v in p_values.items()}
    )
    if explicit is not None and set(explicit) - {str(i) for i in object_ids}:
        raise ValueError("P file includes objects not selected for this bundle")
    targets = {int(i) for i in p_object_ids}
    if explicit is None and targets - {int(i) for i in object_ids}:
        raise ValueError("P objects are not selected for this bundle")
    offsets, audit = {}, {}
    for obj_id in object_ids:
        key = str(obj_id)
        obj_info = info[key]
        diameter = float(obj_info["diameter"])
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("Invalid diameter for obj {}".format(obj_id))
        entry = {"p_target": explicit is None and obj_id in targets}
        if explicit is not None:
            selected = np.asarray(explicit.get(key, [0, 0, 0]), dtype=np.float64)
        elif entry["p_target"]:
            found = symmetry_axis(obj_info)
            if found is None:
                raise ValueError("obj {}: no symmetry axis for P".format(obj_id))
            point, axis, residual = found
            if axis is not None:
                # A continuous axis: the point of the axis nearest the surface centroid
                centroid = surface_centroid(
                    models_path / "obj_{:06d}.ply".format(obj_id)
                )
                point = point + axis * ((centroid - point) @ axis)
            if residual > max_residual_mm:
                raise ValueError(
                    "obj {}: inconsistent symmetry equations ({:.6g} mm)".format(
                        obj_id, residual
                    )
                )
            selected = point
            entry.update(
                symmetry_residual_mm=residual,
                distance_over_diameter=float(np.linalg.norm(point)) / diameter,
            )
        else:
            selected = np.zeros(3)
        if selected.shape != (3,) or not np.isfinite(selected).all():
            raise ValueError(
                "obj {}: P must contain three finite millimeter values".format(obj_id)
            )
        if explicit is not None and np.any(selected):
            found = symmetry_axis(obj_info)
            residual = found[2] if found is not None else 0.0
            if residual > max_residual_mm:
                raise ValueError(
                    "obj {}: inconsistent symmetry equations ({:.6g} mm)".format(
                        obj_id, residual
                    )
                )
        offsets[key] = selected.tolist()
        entry["alignment_applied"] = bool(np.any(selected))
        audit[key] = entry
    return offsets, audit


def shift_model_info(info, p):
    """Convert metadata to the same frame as v' = v-P, preserving other fields."""
    shifted = copy.deepcopy(info)
    p = np.asarray(p, dtype=np.float64)
    for axis, delta in zip("xyz", p):
        if "min_" + axis in shifted:
            shifted["min_" + axis] = float(shifted["min_" + axis] - delta)
    for index, symmetry in enumerate(shifted.get("symmetries_discrete", [])):
        matrix = np.asarray(symmetry, dtype=np.float64).reshape(4, 4).copy()
        matrix[:3, 3] += (matrix[:3, :3] - np.eye(3)) @ p
        shifted["symmetries_discrete"][index] = matrix.reshape(-1).tolist()
    for symmetry in shifted.get("symmetries_continuous", []):
        symmetry["offset"] = (np.asarray(symmetry["offset"]) - p).tolist()
    return shifted


def shift_ply(source, destination, p):
    from plyfile import PlyData

    p = np.asarray(p, dtype=np.float64)
    if not np.any(p):
        shutil.copy2(source, destination)
        return
    ply = PlyData.read(str(source), mmap=False)
    vertex = ply["vertex"].data
    for axis, delta in zip("xyz", p):
        if not np.isfinite(vertex[axis]).all():
            raise ValueError("Non-finite PLY vertices: {}".format(source))
        if not np.issubdtype(vertex[axis].dtype, np.floating):
            raise ValueError("PLY vertex coordinates must be floating point")
        vertex[axis] = vertex[axis] - delta
    ply.write(str(destination))


def keypoints_to_original_frame(cache, offsets):
    """Move a cache generated on v-P meshes back to the original model frame."""
    cache = copy.deepcopy(cache)
    for key, points in cache["keypoints_3d"].items():
        p = np.asarray(offsets[key], dtype=np.float64)
        cache["keypoints_3d"][key] = (np.asarray(points, dtype=np.float64) + p).tolist()
    metadata = cache["metadata"]
    metadata["description"] = "RACE6D 32-point box keypoints, original BOP model frame"
    metadata["coordinate_system"] = "original_bop_model_frame"
    metadata["p_values"] = {k: list(v) for k, v in offsets.items()}
    metadata["training_transform"] = "kp_training = kp_original - P"
    metadata["generation_frame"] = "built on v-P meshes (symmetry box), then +P"
    return cache


def prepare_models(
    models_path,
    output_dir,
    dataset,
    models_eval_path=None,
    object_ids=None,
    p_values=None,
    p_object_ids=None,
    max_residual_mm=0.01,
    keypoint_config=None,
    axis_box=True,
    install_to=None,
):
    """Publish an immutable new bundle: P, original-frame keypoints and hashes.

    Input meshes must be in the ORIGINAL BOP frame. No mesh or models_info is
    written: training reads the original models and applies v-P, kp-P and
    t+R@P exactly once at loading; BOP output reverses it with t-R@P.
    """
    from .race6d_keypoints import generate_cache

    source = Path(models_path).expanduser().resolve()
    eval_source = (
        Path(models_eval_path).expanduser().resolve() if models_eval_path else source
    )
    output = Path(output_dir).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(
            "Output already exists; choose a new directory: {}".format(output)
        )
    for folder in (source, eval_source):
        if folder == output or folder in output.parents:
            raise ValueError(
                "Training output must not be inside an original models folder"
            )
        if any(
            (path / "model_preparation.json").exists()
            for path in (folder, folder.parent)
        ):
            raise ValueError(
                "Already prepared training-frame input; use original BOP meshes"
            )
    info_paths = [source / "models_info.json", eval_source / "models_info.json"]
    infos = [json.loads(path.read_text()) for path in info_paths]
    available = {int(key) for key in infos[1]}
    if object_ids is None:
        if dataset == "lmo":
            object_ids = [1, 5, 6, 8, 9, 10, 11, 12]
        else:
            object_ids = sorted(available)
    object_ids = sorted(set(int(i) for i in object_ids))
    if not object_ids or any(i <= 0 for i in object_ids):
        raise ValueError("Select positive BOP object IDs")
    for folder, info in zip((source, eval_source), infos):
        for obj_id in object_ids:
            if str(obj_id) not in info:
                raise ValueError("models_info is missing obj {}".format(obj_id))
            path = folder / "obj_{:06d}.ply".format(obj_id)
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.with_name(path.name + ".backup_paligned").exists():
                raise ValueError(
                    "Legacy shifted mesh detected: {}; use its original copy".format(
                        path
                    )
                )
    if p_object_ids is None:
        p_object_ids = [i for i in P_OBJECTS.get(dataset, ()) if i in object_ids]
    offsets, audit = find_alignment(
        eval_source, object_ids, p_object_ids, max_residual_mm, p_values
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix="." + output.name + ".", dir=str(output.parent))
    )
    sources, products = {}, {}
    try:
        # Shifted copies exist only in this scratch folder: the keypoint code
        # works in the v-P frame (symmetry box around the axis). They are
        # removed before publishing; meshes on disk stay in the original frame.
        work = stage / ".training_frame"
        work.mkdir()
        for folder, info in zip((source, eval_source), infos):
            sources[str(folder / "models_info.json")] = file_sha256(
                folder / "models_info.json"
            )
            for obj_id in object_ids:
                path = folder / "obj_{:06d}.ply".format(obj_id)
                sources[str(path)] = file_sha256(path)
        moved_info = {}
        for obj_id in object_ids:
            path = eval_source / "obj_{:06d}.ply".format(obj_id)
            shift_ply(path, work / path.name, offsets[str(obj_id)])
            moved_info[str(obj_id)] = shift_model_info(
                infos[1][str(obj_id)], offsets[str(obj_id)]
            )
        (work / "models_info.json").write_text(
            json.dumps(moved_info, indent=2) + "\n"
        )
        cache = generate_cache(
            work,
            object_ids,
            dataset,
            keypoint_config=keypoint_config,
            axis_box=axis_box,
        )
        shutil.rmtree(work)
        cache = keypoints_to_original_frame(cache, offsets)
        (stage / "cached_keypoints_3d_edge.json").write_text(
            json.dumps(cache, indent=2) + "\n"
        )
        (stage / "p_values.json").write_text(json.dumps(offsets, indent=2) + "\n")
        nonzero = {k: v for k, v in offsets.items() if np.any(v)}
        contract = {
            "need_aligned": bool(nonzero),
            "paligned_values": nonzero,
            "aligned_offsets": nonzero,
            "files_frame": "original_bop_model_frame (meshes, GT, keypoint cache)",
            "model_vertex_transform": "v_training = v_original - P (at loading)",
            "keypoint_transform": "kp_training = kp_original - P (at loading)",
            "training_pose_transform": "t_training = t_original + R_ego @ P",
            "bop_output_transform": "t_original = t_training - R_ego @ P",
            "camera_policy": "preserve_original_per_image_cam_K",
            "rotation_frame": "BOP egocentric; allo conversion precedes P reversal",
            "integration_note": "RACE6D reads p_values.json and this cache from coco_path (prepare_race6d_models.py --install_to) and applies v-P, kp-P, t+R@P at loading and t-R@P in the postprocessor; a top-level aligned_offsets in its config overrides the file.",
        }
        (stage / "model_contract.json").write_text(
            json.dumps(contract, indent=2) + "\n"
        )
        for name in (
            "cached_keypoints_3d_edge.json",
            "p_values.json",
            "model_contract.json",
        ):
            products[name] = file_sha256(stage / name)
        summary = {
            "dataset": dataset,
            "object_ids": object_ids,
            "output_dir": str(output),
            "units": "mm",
            "source_hashes": sources,
            "product_hashes": products,
            "alignment_audit": audit,
            "p_selection": "explicit" if p_values is not None else "symmetry_axis",
            "p_object_ids": sorted(int(i) for i in p_object_ids),
            "max_residual_mm": max_residual_mm,
            "axis_box": axis_box,
            "keypoint_config": keypoint_config,
            "source_code_hashes": {
                "race6d_models.py": file_sha256(__file__),
                "race6d_keypoints.py": file_sha256(
                    Path(__file__).with_name("race6d_keypoints.py")
                ),
            },
            "official_bop_data_modified": False,
            "meshes_written": False,
            "evaluation_warning": "Use original BOP meshes/GT for official scoring.",
        }
        (stage / "model_preparation.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        for path, expected in sources.items():
            if file_sha256(path) != expected:
                raise RuntimeError(
                    "Original model input changed during preparation: {}".format(path)
                )
        if os.path.lexists(output):
            raise FileExistsError(
                "Output appeared during preparation: {}".format(output)
            )
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    if install_to is not None:
        summary["installed"] = install_bundle(output, install_to)
    return summary


INSTALLED_FILES = ("cached_keypoints_3d_edge.json", "p_values.json")


def install_bundle(bundle_dir, dataset_dir):
    """Copy the keypoint cache and P of a bundle to a RACE6D coco_path.

    RACE6D reads {coco_path}/cached_keypoints_3d_edge.json and
    {coco_path}/p_values.json, so its configs need no P. A different existing
    file is kept as <name>.before_<timestamp>; an identical one is left as is.
    """
    bundle = Path(bundle_dir).expanduser()
    dataset = Path(dataset_dir).expanduser()
    if not dataset.is_dir():
        raise FileNotFoundError("Dataset directory is missing: {}".format(dataset))
    for name in INSTALLED_FILES:
        if not (bundle / name).is_file():
            raise FileNotFoundError(bundle / name)
    stamp = time.strftime("%Y%m%d%H%M%S")
    report = {}
    for name in INSTALLED_FILES:
        source, target = bundle / name, dataset / name
        entry = {"path": str(target), "sha256": file_sha256(source)}
        if target.exists() and file_sha256(target) == entry["sha256"]:
            entry["action"] = "unchanged"
        else:
            if os.path.lexists(target):
                backup = target.with_name("{}.before_{}".format(name, stamp))
                if os.path.lexists(backup):
                    raise FileExistsError(backup)
                target.rename(backup)
                entry["backup"] = str(backup)
            temp = target.with_name("." + name + ".installing")
            shutil.copyfile(source, temp)
            temp.replace(target)
            entry["action"] = "installed"
        report[name] = entry
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models_path", required=True, help="ORIGINAL BOP models directory"
    )
    parser.add_argument(
        "--models_eval_path",
        help="ORIGINAL models_eval directory; defaults to models_path",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name; selects notebook keypoint presets",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="New training-asset directory; never overwritten",
    )
    parser.add_argument("--object_ids", nargs="+", type=int)
    parser.add_argument(
        "--p_values",
        help="Explicit {obj_id: [Px,Py,Pz]} JSON in mm; {} disables shifts",
    )
    parser.add_argument(
        "--p_object_ids",
        nargs="*",
        type=int,
        help="Objects that get the symmetry-axis P (default: YCB-V 19 20, ITODD 2; "
        "none for other datasets); the others keep P=0",
    )
    parser.add_argument("--max_residual_mm", type=float, default=0.01)
    parser.add_argument(
        "--keypoint_config",
        help="JSON with split_config / edge_override object mappings",
    )
    parser.add_argument(
        "--no_axis_box", action="store_false", dest="axis_box", default=True
    )
    parser.add_argument(
        "--install_to",
        help="RACE6D coco_path (dataset folder): copy the keypoint cache and "
        "p_values.json there; a different existing file is kept as "
        "<name>.before_<timestamp>",
    )
    args = vars(parser.parse_args(argv))
    try:
        for key in ("p_values", "keypoint_config"):
            if args[key] is not None:
                args[key] = json.loads(Path(args[key]).expanduser().read_text())
        result = prepare_models(**args)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        parser.exit(1, "RACE6D model preparation failed: {}\n".format(error))
    print(json.dumps(result, indent=2))
    return result
