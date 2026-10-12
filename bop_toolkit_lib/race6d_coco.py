"""Prepare a complete RACE6D COCO split without modifying BOP source files.

Only one scene's source JSON and one object's masks are held by each worker.
Workers spool JSON lines to disk; the final COCO arrays are merged as streams.
The output directory is published only after every scene succeeds.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from bop_toolkit_lib import config, dataset_params, inout, pycoco_utils
from bop_toolkit_lib.race6d_rotations import (
    DEFAULT_MAX_CHANGE, DEFAULT_TOLERANCE, file_sha256, prepare_rotation,
    rotation_metrics, validate_rotation_options,
)


def _json_line(handle, value):
    handle.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def _require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Required BOP input is missing: {}".format(path))
    return path


def _image_path(path):
    """Respect actual image extensions in repackaged BOP downloads."""
    path = Path(path)
    if path.is_file():
        return path
    alternatives = [path.with_suffix(ext) for ext in
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                    if path.with_suffix(ext).is_file()]
    if len(alternatives) == 1:
        return alternatives[0]
    if len(alternatives) > 1:
        raise ValueError("Ambiguous image extensions for {}".format(path))
    return _require_file(path)


def _transfer(source, destination, mode):
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        # Absolute links remain valid when the staging directory is renamed.
        destination.symlink_to(source.resolve())


def _transfer_depth(source, destination, mode):
    """RACE6D resolves depth by replacing the RGB extension with .png."""
    if source.suffix.lower() == ".png":
        _transfer(source, destination, mode)
    else:
        depth = inout.load_depth(source)
        if (not np.isfinite(depth).all() or np.any(depth < 0)
                or np.any(depth > 65535) or np.any(depth != np.rint(depth))):
            raise ValueError("Depth cannot be represented losslessly as uint16 PNG: {}".format(source))
        inout.save_depth(destination, depth)


def _scene_paths(dp_split, scene_id):
    keys = dataset_params.scene_tpaths_keys(
        dp_split["eval_modality"], dp_split["eval_sensor"], scene_id
    )
    # ITODD's real images are grayscale TIFFs; its PBR training images are RGB.
    if (dp_split["eval_sensor"] is None
            and "rgb" not in dp_split["im_modalities"]
            and dp_split["split_type"] != "pbr"):
        keys["rgb_tpath"] = "gray_tpath"
    return {key: value.replace("{scene_id:06d}", "{:06d}".format(scene_id))
            for key, template_key in keys.items()
            if (value := dp_split.get(template_key)) is not None}


def _process_scene(job):
    """Worker entry point; return counts/paths, never full annotations."""
    paths = job["paths"]
    scene_id = job["scene_id"]
    scene_gt = inout.load_json(paths["scene_gt_tpath"], keys_to_int=True)
    scene_info = inout.load_json(paths["scene_gt_info_tpath"], keys_to_int=True)
    scene_camera = inout.load_json(paths["scene_camera_tpath"], keys_to_int=True)
    spool = Path(job["staging"]) / ".spool"
    image_spool = spool / "{:06d}.images.jsonl".format(scene_id)
    annotation_spool = spool / "{:06d}.annotations.jsonl".format(scene_id)
    rotation_spool = spool / "{:06d}.rotations.jsonl".format(scene_id)
    image_root = Path(job["staging"]) / job["coco_split"]
    depth_root = Path(job["staging"]) / (job["coco_split"] + "_depth")
    annotation_count = 0
    skipped_instances = 0
    rotation_audit = {"checked": 0, "normalized": 0, "max_entry_change": 0.0,
                      "max_source_orthogonality_error": 0.0}

    with image_spool.open("w") as images_out, annotation_spool.open("w") as annotations_out, rotation_spool.open("w") as rotations_out:
        for offset, im_id in enumerate(job["image_ids"]):
            image_id = job["image_start"] + offset
            instances = scene_gt[im_id]
            if im_id not in scene_info or im_id not in scene_camera:
                raise ValueError("Scene {} image {} lacks GT info or camera data".format(scene_id, im_id))
            gt_info = scene_info[im_id]
            camera = scene_camera[im_id]
            if len(gt_info) != len(instances):
                raise ValueError("Scene {} image {}: GT/GT-info instance counts differ".format(scene_id, im_id))
            cam_k = np.asarray(camera.get("cam_K", []), dtype=float).reshape(-1)
            if cam_k.size != 9 or not np.isfinite(cam_k).all():
                raise ValueError("Scene {} image {} requires a finite pinhole cam_K (9 values)".format(scene_id, im_id))
            matrix = cam_k.reshape(3, 3)
            if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                    or not np.allclose(matrix[2], [0, 0, 1], atol=1e-8, rtol=0)):
                raise ValueError("Scene {} image {} has invalid pinhole cam_K: positive fx/fy and bottom row [0,0,1] required".format(scene_id, im_id))
            # Keep the actual per-image K. Different train/test cameras and
            # cropped principal points are legitimate, not normalization errors.
            depth_scale = float(camera.get("depth_scale", 1.0))
            if not np.isfinite(depth_scale) or depth_scale <= 0:
                raise ValueError("Scene {} image {} has invalid depth_scale".format(scene_id, im_id))

            source_image = _image_path(paths["rgb_tpath"].format(im_id=im_id))
            with Image.open(source_image) as image:
                width, height = image.size
            image_name = "{:06d}{}".format(image_id, source_image.suffix)
            _transfer(source_image, image_root / image_name, job["image_mode"])
            image_info = {
                "id": image_id, "file_name": image_name,
                "width": width, "height": height,
                "scene_id": "{:06d}".format(scene_id),
                "original_filename": source_image.name,
                "bop_im_id": im_id,
            }
            if job["include_depth"]:
                if "depth_tpath" not in paths:
                    raise ValueError("Selected dataset/sensor has no depth modality")
                source_depth = _image_path(paths["depth_tpath"].format(im_id=im_id))
                with Image.open(source_depth) as depth_image:
                    if depth_image.size != (width, height):
                        raise ValueError("RGB/depth sizes differ for {}".format(source_image))
                _transfer_depth(source_depth, depth_root / "{:06d}.png".format(image_id), job["image_mode"])
            _json_line(images_out, image_info)

            for gt_id, instance in enumerate(instances):
                obj_id = int(instance["obj_id"])
                if obj_id not in job["obj_ids"]:
                    # LM-O keeps the original eight BOP category IDs.
                    continue
                visibility = float(gt_info[gt_id]["visib_fract"])
                if not np.isfinite(visibility) or not 0 <= visibility <= 1:
                    raise ValueError("Invalid visib_fract in scene {} image {} GT {}".format(scene_id, im_id, gt_id))
                mask_visib_path = _require_file(paths["mask_visib_tpath"].format(im_id=im_id, gt_id=gt_id))
                visible = inout.load_depth(mask_visib_path).astype(bool)
                if visible.shape != (height, width):
                    raise ValueError("Mask/image sizes differ: {}".format(mask_visib_path))
                if not visible.any():
                    skipped_instances += 1
                    continue
                mask_full_path = _require_file(paths["mask_tpath"].format(im_id=im_id, gt_id=gt_id))
                full = inout.load_depth(mask_full_path).astype(bool)
                if full.shape != visible.shape:
                    raise ValueError("Visible/full mask sizes differ: {}".format(mask_full_path))
                if not full.any():
                    skipped_instances += 1
                    continue
                bbox_mask = full if job["bbox_type"] == "amodal" else visible
                bbox = pycoco_utils.bbox_from_binary_mask(bbox_mask)
                rotation = np.asarray(instance["cam_R_m2c"], dtype=np.float64)
                metrics, changed = None, False
                if job["rotation_mode"] != "preserve":
                    try:
                        rotation, metrics, changed = prepare_rotation(
                            rotation, job["rotation_mode"], job["rotation_tolerance"],
                            job["rotation_max_change"])
                    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
                        raise ValueError("{}: image {} GT {} object {}: {}".format(
                            paths["scene_gt_tpath"], im_id, gt_id, obj_id, error)) from error
                rotation = rotation.reshape(3, 3)
                translation = np.asarray(instance["cam_t_m2c"], dtype=np.float64).reshape(3)
                pose = np.concatenate((translation, rotation.reshape(-1))).tolist()
                annotation = pycoco_utils.create_annotation_info(
                    annotation_count + 1, image_id, obj_id,
                    binary_mask=visible, bounding_box=bbox,
                    binary_mask_full=full, poses=pose, cam_K=cam_k.tolist(),
                    depth_scale=depth_scale, visibility=visibility,
                    ignore=visibility < 0.1,
                )
                if annotation is not None:
                    _json_line(annotations_out, annotation)
                    if metrics is not None:
                        rotation_audit["checked"] += 1
                        rotation_audit["max_source_orthogonality_error"] = max(
                            rotation_audit["max_source_orthogonality_error"], metrics["orthogonality_error"])
                    if changed:
                        change = float(np.max(np.abs(rotation.reshape(-1)
                                                     - np.asarray(instance["cam_R_m2c"]).reshape(-1))))
                        rotation_audit["normalized"] += 1
                        rotation_audit["max_entry_change"] = max(rotation_audit["max_entry_change"], change)
                        _json_line(rotations_out, {
                            "scene_id": scene_id, "bop_im_id": im_id, "gt_id": gt_id,
                            "obj_id": obj_id, "image_id": image_id,
                            "original_rotation": instance["cam_R_m2c"],
                            "normalized_rotation": rotation.reshape(-1).tolist(),
                            "max_entry_change": change, "source_metrics": metrics,
                            "normalized_metrics": rotation_metrics(rotation, job["rotation_tolerance"]),
                        })
                    annotation_count += 1
    result = {"scene_id": scene_id, "images": len(job["image_ids"]),
              "annotations": annotation_count, "skipped_instances": skipped_instances,
              "image_spool": str(image_spool), "annotation_spool": str(annotation_spool),
              "rotation_spool": str(rotation_spool), "rotation_audit": rotation_audit}
    if job["rotation_mode"] != "preserve":
        result["rotation_source"] = {"scene_id": scene_id,
                                     "path": str(Path(paths["scene_gt_tpath"]).resolve()),
                                     "sha256": file_sha256(paths["scene_gt_tpath"])}
    return result


def _merge_spools(path, metadata, results):
    with path.open("w") as output:
        output.write("{")
        for key, value in metadata.items():
            output.write(json.dumps(key) + ":" + json.dumps(value, allow_nan=False) + ",\n")
        for array_index, (array_name, spool_name) in enumerate(
                (("images", "image_spool"), ("annotations", "annotation_spool"))):
            if array_index:
                output.write(",\n")
            output.write(json.dumps(array_name) + ":[\n")
            count = 0
            for result in results:
                with Path(result[spool_name]).open() as source:
                    for line in source:
                        if count:
                            output.write(",\n")
                        count += 1
                        if array_name == "annotations":
                            annotation = json.loads(line)
                            annotation["id"] = count
                            output.write(json.dumps(annotation, separators=(",", ":"), allow_nan=False))
                        else:
                            output.write(line.rstrip("\n"))
            output.write("\n]")
        output.write("\n}\n")


def prepare_dataset(dataset="lmo", dataset_split="test", dataset_split_type=None,
                    datasets_path=None, bbox_type="amodal", output_dir=None,
                    targets_filename=None, use_all_gt=False,
                    scene_ids=None, image_ids=None, max_images=None,
                    num_workers=1, image_mode="symlink", include_depth=False,
                    rotation_mode="preserve", rotation_tolerance=DEFAULT_TOLERANCE,
                    rotation_max_change=DEFAULT_MAX_CHANGE):
    """Convert selected BOP images and return paths/counts for the finished split.

    ``dataset_split_type=None`` means PBR for train and the official Toolkit
    default sensor/type for test/val. Use ``"none"`` for an unsuffixed train split.
    Test uses the target image list (``test_targets_bop19.json`` by default). Val
    uses all GT images unless ``targets_filename`` is given: BOP val splits have
    no target list, and they are the annotated split for datasets whose test GT
    is not public (HB, ITODD). Train always uses all available GT images.
    ``image_ids`` filters original BOP image IDs in every selected scene.
    Existing output directories are always refused, including empty directories.
    Rotation checks apply to exported annotations, not empty/excluded instances.
    ``normalize`` projects only slightly invalid rotations in the new output;
    source GT is never rewritten. Default ``preserve`` keeps historical behavior.
    """
    if bbox_type not in ("modal", "amodal"):
        raise ValueError("bbox_type must be modal or amodal")
    if image_mode not in ("symlink", "copy"):
        raise ValueError("image_mode must be symlink or copy")
    if num_workers < 0 or (max_images is not None and max_images < 1):
        raise ValueError("num_workers must be nonnegative and max_images positive")
    validate_rotation_options(rotation_mode, rotation_tolerance, rotation_max_change)
    root = Path(datasets_path or config.datasets_path).expanduser().resolve()
    split_type = dataset_split_type
    if split_type is None and dataset_split == "train":
        split_type = "pbr"
    elif isinstance(split_type, str) and split_type.lower() in ("none", "null", ""):
        split_type = None
    try:
        dp_split = dataset_params.get_split_params(str(root), dataset, dataset_split, split_type)
        dp_model = dataset_params.get_model_params(str(root), dataset)
    except (KeyError, ValueError, AssertionError) as error:
        raise ValueError("Unsupported BOP dataset/split/type: {}/{}/{} ({})".format(
            dataset, dataset_split, split_type, error)) from error
    split_path = Path(dp_split["split_path"])
    if not split_path.is_dir():
        raise FileNotFoundError("BOP split directory is missing: {}. Set --datasets_path or --dataset_split_type.".format(split_path))
    if not isinstance(dp_split["im_modalities"], list):
        raise ValueError("RACE6D conversion currently supports classic BOP datasets with pinhole cameras; {} needs a dedicated sensor/camera conversion".format(dataset))

    if targets_filename is None and dataset_split == "test":
        targets_filename = "test_targets_bop19.json"
    all_gt = use_all_gt or dataset_split == "train" or targets_filename is None
    targets = None
    if not all_gt:
        target_path = Path(targets_filename).expanduser()
        if not target_path.is_absolute():
            target_path = Path(dp_split["base_path"]) / target_path
        targets = {}
        for entry in inout.load_json(_require_file(target_path)):
            targets.setdefault(int(entry["scene_id"]), set()).add(int(entry["im_id"]))
    present_ids = {int(path.name) for path in split_path.iterdir()
                   if path.is_dir() and path.name.isdecimal()}
    selected_scenes = sorted(set(scene_ids) if scene_ids is not None else
                             (set(targets) if targets is not None else present_ids))
    missing = set(selected_scenes) - present_ids
    if missing:
        raise FileNotFoundError("BOP scenes {} are missing from {}".format(sorted(missing), split_path))
    if not selected_scenes:
        raise ValueError("No scenes selected from {}".format(split_path))
    selected_images = set(image_ids) if image_ids is not None else None
    coco_split = "train2017" if dataset_split == "train" else "val2017"
    split_label = split_path.name if all_gt else dataset_split + "_eval"
    default_name = "{}2coco_{}_{}".format(dataset, split_label, bbox_type)
    output = (Path(output_dir).expanduser() if output_dir is not None
              else Path(dp_split["base_path"]) / default_name).absolute()
    if os.path.lexists(output):
        raise FileExistsError("Output already exists; choose a new --output_dir: {}".format(output))

    jobs = []
    total_images = 0
    for scene_id in selected_scenes:
        paths = _scene_paths(dp_split, scene_id)
        for key in ("scene_gt_tpath", "scene_gt_info_tpath", "scene_camera_tpath"):
            _require_file(paths[key])
        # Load only one source scene at a time; never load prior merged COCO files.
        gt = inout.load_json(paths["scene_gt_tpath"], keys_to_int=True)
        ids = set(gt)
        del gt
        if targets is not None:
            required = targets.get(scene_id, set())
            if selected_images is not None:
                required = required & selected_images
            if required - ids:
                raise ValueError("Target images {} have no GT in scene {}".format(sorted(required - ids), scene_id))
            ids &= required
        if selected_images is not None:
            ids &= selected_images
        ids = sorted(ids)
        if max_images is not None:
            ids = ids[:max_images - total_images]
        if ids:
            jobs.append({"scene_id": scene_id, "paths": paths, "image_ids": ids,
                         "image_start": total_images, "bbox_type": bbox_type,
                         "obj_ids": set(dp_model["obj_ids"]), "coco_split": coco_split,
                         "image_mode": image_mode, "include_depth": include_depth,
                         "rotation_mode": rotation_mode, "rotation_tolerance": rotation_tolerance,
                         "rotation_max_change": rotation_max_change})
            total_images += len(ids)
        if max_images is not None and total_images >= max_images:
            break
    if not total_images:
        raise ValueError("No images match the requested scenes, targets and image IDs")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".tmp-", dir=output.parent))
    try:
        (staging / coco_split).mkdir()
        (staging / "annotations").mkdir()
        (staging / ".spool").mkdir()
        if include_depth:
            (staging / (coco_split + "_depth")).mkdir()
        for job in jobs:
            job["staging"] = str(staging)
        worker_count = min(num_workers or os.cpu_count() or 1, len(jobs))
        if worker_count == 1:
            results = [_process_scene(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                results = list(pool.map(_process_scene, jobs))
        metadata = {
            "info": {
                "description": "{} {} RACE6D {} annotations".format(dataset, split_path.name, bbox_type),
                "version": "1.0", "date_created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "bop_split": split_path.name, "bbox_type": bbox_type,
                "selection": "all_gt" if all_gt else str(targets_filename),
            },
            "licenses": [],
            "categories": [
                {"id": obj_id, "name": str(obj_id),
                 "supercategory": "symmetric" if obj_id in dp_model["symmetric_obj_ids"] else "non-symmetric"}
                for obj_id in sorted(dp_model["obj_ids"])
            ],
        }
        annotation_name = "instances_{}.json".format(coco_split)
        _merge_spools(staging / "annotations" / annotation_name, metadata, results)
        if rotation_mode != "preserve":
            with (staging / "rotation_changes.jsonl").open("w") as changes:
                for result in results:
                    with Path(result["rotation_spool"]).open() as source:
                        shutil.copyfileobj(source, changes)
        shutil.rmtree(staging / ".spool")
        summary = {
            "output_dir": str(output), "image_dir": str(output / coco_split),
            "annotation_file": str(output / "annotations" / annotation_name),
            "scenes": len(results), "images": total_images,
            "annotations": sum(result["annotations"] for result in results),
            "skipped_empty_instances": sum(result["skipped_instances"] for result in results),
            "image_mode": image_mode,
        }
        summary["rotations"] = {
            "mode": rotation_mode, "tolerance": rotation_tolerance,
            "max_allowed_entry_change": rotation_max_change,
            "scope": "exported_annotations_only",
            "checked": sum(r["rotation_audit"]["checked"] for r in results),
            "normalized": sum(r["rotation_audit"]["normalized"] for r in results),
            "max_entry_change": max(r["rotation_audit"]["max_entry_change"] for r in results),
            "max_source_orthogonality_error": max(
                r["rotation_audit"]["max_source_orthogonality_error"] for r in results),
        }
        if rotation_mode != "preserve":
            summary["rotations"]["changes_file"] = str(output / "rotation_changes.jsonl")
            summary["rotations"]["sources"] = [r["rotation_source"] for r in results]
        if include_depth:
            summary["depth_dir"] = str(output / (coco_split + "_depth"))
        with (staging / "conversion_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        if os.path.lexists(output):
            raise FileExistsError("Output appeared during conversion; refusing to overwrite: {}".format(output))
        staging.rename(output)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parse_ids(values):
    if values is None:
        return None
    try:
        return sorted({int(item) for value in values for item in value.split(",")})
    except ValueError as error:
        raise ValueError("Scene/image IDs must be integers separated by spaces or commas") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="lmo")
    parser.add_argument("--dataset_split", "--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--dataset_split_type", "--split_type", help="Default: pbr for train; official sensor/type for test. Use none for an unsuffixed split.")
    parser.add_argument("--datasets_path", default=config.datasets_path, help="Parent of dataset folders (default: BOP_PATH or ~/Downloads)")
    parser.add_argument("--bbox_type", choices=("modal", "amodal"), default="amodal")
    parser.add_argument("--output_dir", help="New output directory; an existing path is never overwritten")
    parser.add_argument("--targets_filename", help="Image list relative to the dataset folder (default: test_targets_bop19.json for test; val uses all GT images)")
    parser.add_argument("--use_all_gt", action="store_true", help="Use all GT images instead of evaluation targets (default for train and val)")
    parser.add_argument("--scene_ids", nargs="+", help="Original BOP scene IDs, space/comma separated")
    parser.add_argument("--image_ids", nargs="+", help="Original BOP image IDs in each scene, space/comma separated")
    parser.add_argument("--max_images", "--max-images", type=int, help="Maximum images across all selected scenes")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel scene workers; 0 = CPU count")
    parser.add_argument("--image_mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--include_depth", action="store_true", help="Also prepare train2017_depth/val2017_depth PNGs")
    parser.add_argument("--rotation_mode", choices=("preserve", "check", "normalize"), default="preserve",
                        help="preserve: unchanged poses; check: reject invalid rotations; normalize: bounded SO(3) projection in NEW output only")
    parser.add_argument("--rotation_tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--rotation_max_change", type=float, default=DEFAULT_MAX_CHANGE,
                        help="Maximum absolute change to any rotation entry when normalizing (default: 0.02, not an angle)")
    args = vars(parser.parse_args(argv))
    try:
        args["scene_ids"] = _parse_ids(args["scene_ids"])
        args["image_ids"] = _parse_ids(args["image_ids"])
        summary = prepare_dataset(**args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "RACE6D COCO conversion failed: {}\n".format(error))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
