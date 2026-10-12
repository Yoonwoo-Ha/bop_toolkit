"""Read-only BOP rotation audit and bounded SO(3) projection for COCO export.

No function in this module rewrites scene_gt.json. Projection uses all nine
matrix entries in float64; it is not the legacy float32 6D reconstruction.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_TOLERANCE = 1e-6
DEFAULT_MAX_CHANGE = 2e-2


def validate_rotation_options(mode, tolerance, max_change):
    if mode not in ("preserve", "check", "normalize"):
        raise ValueError("rotation_mode must be preserve, check or normalize")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("rotation_tolerance must be finite and positive")
    if not np.isfinite(max_change) or max_change <= 0:
        raise ValueError("rotation_max_change must be finite and positive")


def rotation_matrix(value):
    """Accept BOP's nine row-major values or a 3x3 array, without mutating it."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape not in ((9,), (3, 3)):
        raise ValueError("cam_R_m2c must contain nine row-major values or a 3x3 matrix")
    matrix = matrix.reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError("cam_R_m2c contains non-finite values")
    return matrix


def rotation_metrics(value, tolerance=DEFAULT_TOLERANCE):
    """Check determinant, orthogonality, row/column norms and trace bounds."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("rotation_tolerance must be finite and positive")
    matrix = rotation_matrix(value)
    with np.errstate(over="ignore", invalid="ignore"):
        determinant = float(np.linalg.det(matrix))
        trace = float(np.trace(matrix))
        orthogonality = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
        row_error = float(np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1)))
        column_error = float(np.max(np.abs(np.linalg.norm(matrix, axis=0) - 1)))
    if not np.isfinite([determinant, trace, orthogonality, row_error, column_error]).all():
        raise ValueError("cam_R_m2c is too large for a finite rotation audit")
    return {
        "determinant": determinant,
        "determinant_error": abs(determinant - 1),
        "orthogonality_error": orthogonality,
        "row_norm_error": row_error,
        "column_norm_error": column_error,
        "trace": trace,
        "valid": bool(abs(determinant - 1) <= tolerance
                      and orthogonality <= tolerance
                      and row_error <= tolerance and column_error <= tolerance
                      and -1 - tolerance <= trace <= 3 + tolerance),
    }


def prepare_rotation(value, mode="check", tolerance=DEFAULT_TOLERANCE,
                     max_change=DEFAULT_MAX_CHANGE):
    """Return (matrix, source_metrics, changed); refuse grossly invalid labels.

    Matrices already within tolerance are returned unchanged, even in normalize
    mode. Invalid matrices are repaired only if det > 0 and the largest change
    to any matrix entry is <= max_change. This bound is not an angular error.
    """
    validate_rotation_options(mode, tolerance, max_change)
    matrix = rotation_matrix(value)
    if mode == "preserve":
        return matrix, None, False
    metrics = rotation_metrics(matrix, tolerance)
    if metrics["valid"]:
        return matrix, metrics, False
    if mode == "check":
        raise ValueError("Invalid rotation: det={:.9g}, orthogonality={:.9g}, "
                         "row/column norm error={:.9g}/{:.9g}, trace={:.9g}".format(
                             metrics["determinant"], metrics["orthogonality_error"],
                             metrics["row_norm_error"], metrics["column_norm_error"],
                             metrics["trace"]))
    if metrics["determinant"] <= 0:
        raise ValueError("Refusing rotation normalization: reflection or singular matrix (det <= 0)")
    left, _, right = np.linalg.svd(matrix)
    normalized = left @ right
    change = float(np.max(np.abs(normalized - matrix)))
    if change > max_change:
        raise ValueError("Refusing rotation normalization: max entry change {:.9g} exceeds {:.9g}".format(
            change, max_change))
    if not rotation_metrics(normalized, tolerance)["valid"]:
        raise ValueError("SO(3) projection failed the requested rotation tolerance")
    return normalized, metrics, True


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_scene_gt(path, tolerance=DEFAULT_TOLERANCE, max_examples=20):
    """Inspect every instance of one raw BOP scene, including invisible objects."""
    if max_examples < 0:
        raise ValueError("max_examples must be nonnegative")
    validate_rotation_options("check", tolerance, DEFAULT_MAX_CHANGE)
    path = Path(path)
    content = path.read_bytes()
    gt = json.loads(content)
    result = {
        "source": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest(),
        "instances": 0, "invalid": 0, "examples": [],
        "max_determinant_error": 0.0, "max_orthogonality_error": 0.0,
        "max_row_norm_error": 0.0, "max_column_norm_error": 0.0,
    }
    for im_id in sorted(gt, key=int):
        for gt_id, instance in enumerate(gt[im_id]):
            result["instances"] += 1
            try:
                metrics = rotation_metrics(instance["cam_R_m2c"], tolerance)
                for key in ("determinant_error", "orthogonality_error",
                            "row_norm_error", "column_norm_error"):
                    result["max_" + key] = max(result["max_" + key], metrics[key])
                if metrics["valid"]:
                    continue
                details = {"metrics": metrics}
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
                details = {"error": str(error)}
            result["invalid"] += 1
            if len(result["examples"]) < max_examples:
                result["examples"].append({"bop_im_id": int(im_id), "gt_id": gt_id,
                                           "obj_id": instance.get("obj_id"), **details})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only audit of raw BOP scene_gt.json rotations")
    parser.add_argument("--input", required=True, help="One scene_gt.json, one scene directory, or one BOP split directory")
    parser.add_argument("--report", help="Optional NEW JSON report path; never overwrites a file")
    parser.add_argument("--rotation_tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--max_examples", type=int, default=20, help="Maximum invalid examples per scene")
    args = parser.parse_args(argv)
    try:
        source = Path(args.input).expanduser()
        if source.is_file():
            paths = [source]
        elif (source / "scene_gt.json").is_file():
            paths = [source / "scene_gt.json"]
        elif source.is_dir():
            paths = sorted(source.glob("*/scene_gt.json"))
        else:
            raise FileNotFoundError("Input does not exist: {}".format(source))
        if not paths:
            raise ValueError("No scene_gt.json found; pass a raw BOP split, not a merged COCO folder")
        report_path = Path(args.report).expanduser() if args.report else None
        if report_path is not None and (report_path.exists() or report_path.is_symlink()):
            raise FileExistsError("Report already exists: {}".format(report_path))
        results = [audit_scene_gt(path, args.rotation_tolerance, args.max_examples) for path in paths]
        report = {"mode": "check_only", "rotation_tolerance": args.rotation_tolerance,
                  "scenes": len(results), "instances": sum(r["instances"] for r in results),
                  "invalid": sum(r["invalid"] for r in results), "scene_results": results}
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("x") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        parser.exit(1, "RACE6D rotation audit failed: {}\n".format(error))
    if report["invalid"]:
        parser.exit(1, "Found {} invalid rotations; source files were not modified.\n".format(report["invalid"]))
    return report
