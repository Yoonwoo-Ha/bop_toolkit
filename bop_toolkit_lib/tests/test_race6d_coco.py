"""Regression checks for upstream annotation compatibility and RACE6D preparation."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from bop_toolkit_lib import pycoco_utils
from bop_toolkit_lib.race6d_coco import prepare_dataset


VISIBLE = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
FULL = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=bool)
POSE = [10, 20, 600, 0, -1, 0, 1, 0, 0, 0, 0, 1]
CAMERA = [500, 0, 2, 0, 510, 1.5, 0, 0, 1]


def test_upstream_simple_api_and_positional_tail():
    expected = {
        "id": 1, "image_id": 2, "category_id": 5, "iscrowd": 0,
        "area": 2, "bbox": [1, 0, 1, 2],
        "segmentation": {"counts": [3, 2, 7], "size": [3, 4]},
        "width": 4, "height": 3, "ignore": True,
    }
    keyword = pycoco_utils.create_annotation_info(
        annotation_id=1, image_id=2, object_id=5, binary_mask=VISIBLE,
        bounding_box=[1, 0, 1, 2], ignore=True,
    )
    positional = pycoco_utils.create_annotation_info(
        1, 2, 5, VISIBLE, [1, 0, 1, 2], "rle", 0, True,
    )
    assert keyword == positional == expected
    assert pycoco_utils.create_annotation_info(1, 2, 5, VISIBLE * 0, [0, 0, 0, 0]) is None


@pytest.mark.parametrize("style", ["keywords", "rich", "middle", "legacy_alias"])
def test_rich_and_legacy_annotations(style):
    kwargs = dict(visibility=0.08, cam_K=CAMERA, depth_scale=0.1)
    if style == "rich":
        annotation = pycoco_utils.create_annotation_info(
            1, 2, 5, VISIBLE, FULL, [0, 0, 3, 2], POSE, "rle", 0, True, **kwargs)
    elif style == "middle":
        annotation = pycoco_utils.create_annotation_info(
            1, 2, 5, VISIBLE, np.array([0, 0, 3, 2]), POSE,
            binary_mask_full=FULL, ignore=True, **kwargs)
    else:
        mask_key = "binary_mask_visib" if style == "legacy_alias" else "binary_mask"
        annotation = pycoco_utils.create_annotation_info(
            1, 2, 5, **{mask_key: VISIBLE}, binary_mask_full=FULL,
            bounding_box=[0, 0, 3, 2], poses=POSE, ignore=True, **kwargs)
    assert annotation["pose"] == POSE
    assert annotation["cam_K"] == CAMERA
    assert annotation["depth_scale"] == 0.1
    assert annotation["visibility"] == 0.08
    assert annotation["ignore"] is True
    assert annotation["area"] == 2  # visible area even with amodal bbox
    assert annotation["segmentation"] == {"counts": [3, 2, 7], "size": [3, 4]}
    assert annotation["full_segmentation"] == {"counts": [0, 2, 1, 2, 1, 2, 4], "size": [3, 4]}


def test_polygon_tail_keeps_tolerance_and_ignore():
    positional = pycoco_utils.create_annotation_info(
        1, 2, 5, FULL, [0, 0, 3, 2], "polygon", 0, True)
    keyword = pycoco_utils.create_annotation_info(
        1, 2, 5, FULL, bounding_box=[0, 0, 3, 2],
        mask_encoding_format="polygon", tolerance=0, ignore=True)
    assert positional == keyword
    assert positional["ignore"] is True
    assert positional["segmentation"]
    assert "pose" not in positional and "full_segmentation" not in positional


def _write_json(path, value):
    path.write_text(json.dumps(value))


def _write_scene(root, dataset, split, scene_id, image_ids=(3, 8, 13),
                 extension=".png", depth_scale=None, gray=False):
    scene = root / dataset / split / "{:06d}".format(scene_id)
    for folder in (("gray" if gray else "rgb"), "depth", "mask", "mask_visib"):
        (scene / folder).mkdir(parents=True)
    gt, info, camera = {}, {}, {}
    for im_id in image_ids:
        image_name = "{:06d}{}".format(im_id, extension)
        Image.fromarray(np.full((3, 4, 3), 128, dtype=np.uint8)).save(scene / ("gray" if gray else "rgb") / image_name)
        depth = np.arange(12, dtype=np.uint16).reshape(3, 4) * 100
        depth_extension = ".tif" if gray else ".png"
        Image.fromarray(depth).save(scene / "depth" / "{:06d}{}".format(im_id, depth_extension))
        gt[str(im_id)] = [
            {"obj_id": 5, "cam_t_m2c": POSE[:3], "cam_R_m2c": POSE[3:]},
            {"obj_id": 1, "cam_t_m2c": POSE[:3], "cam_R_m2c": POSE[3:]},
        ]
        info[str(im_id)] = [{"visib_fract": 0.08}, {"visib_fract": 0.0}]
        camera[str(im_id)] = {"cam_K": CAMERA}
        if depth_scale is not None:
            camera[str(im_id)]["depth_scale"] = depth_scale
        for gt_id in (0, 1):
            mask_name = "{:06d}_{:06d}.png".format(im_id, gt_id)
            Image.fromarray((FULL * 255).astype(np.uint8)).save(scene / "mask" / mask_name)
            mask = VISIBLE if gt_id == 0 else VISIBLE * 0
            Image.fromarray((mask * 255).astype(np.uint8)).save(scene / "mask_visib" / mask_name)
    _write_json(scene / "scene_gt.json", gt)
    _write_json(scene / "scene_gt_info.json", info)
    _write_json(scene / "scene_camera.json", camera)
    return scene


def _load_output(summary):
    return json.loads(Path(summary["annotation_file"]).read_text())


def test_test_targets_complete_layout_and_source_unchanged(tmp_path):
    scene = _write_scene(tmp_path, "lmo", "test", 2)
    target = tmp_path / "lmo" / "test_targets_bop19.json"
    _write_json(target, [{"scene_id": 2, "im_id": 8, "obj_id": 5, "inst_count": 1},
                         {"scene_id": 2, "im_id": 3, "obj_id": 5, "inst_count": 1}])
    before = {p.relative_to(scene): p.read_bytes() for p in scene.rglob("*") if p.is_file()}
    summary = prepare_dataset(datasets_path=tmp_path, output_dir=tmp_path / "output", include_depth=True)
    data = _load_output(summary)
    assert summary["images"] == 2 and summary["annotations"] == 2
    assert summary["skipped_empty_instances"] == 2
    assert [image["bop_im_id"] for image in data["images"]] == [3, 8]
    assert [image["id"] for image in data["images"]] == [0, 1]
    assert data["images"][0]["scene_id"] == "000002"
    assert data["images"][0]["original_filename"] == "000003.png"
    assert data["images"][0]["width"] == 4
    assert data["annotations"][0]["bbox"] == [0, 0, 3, 2]
    assert data["annotations"][0]["depth_scale"] == 1.0
    assert data["annotations"][0]["pose"] == POSE
    assert data["annotations"][0]["cam_K"] == CAMERA
    assert data["annotations"][0]["ignore"] is True
    assert [c["id"] for c in data["categories"]] == [1, 5, 6, 8, 9, 10, 11, 12]
    assert next(c for c in data["categories"] if c["id"] == 10)["supercategory"] == "symmetric"
    assert (tmp_path / "output/val2017/000000.png").is_symlink()
    assert (tmp_path / "output/val2017_depth/000000.png").resolve() == scene / "depth/000003.png"
    assert not (tmp_path / "output/.spool").exists()
    assert before == {p.relative_to(scene): p.read_bytes() for p in scene.rglob("*") if p.is_file()}


def test_parallel_pbr_stable_merge_and_actual_jpeg_extensions(tmp_path):
    for scene_id in (1, 0):
        _write_scene(tmp_path, "ycbv", "train_pbr", scene_id, image_ids=(8, 3), extension=".jpg", depth_scale=0.1)
    outputs = []
    for workers in (1, 2):
        summary = prepare_dataset(dataset="ycbv", dataset_split="train", datasets_path=tmp_path,
                                  output_dir=tmp_path / str(workers), num_workers=workers,
                                  image_mode="copy", bbox_type="modal", include_depth=True)
        data = _load_output(summary)
        assert summary["images"] == 4 and summary["annotations"] == 4
        assert [ann["id"] for ann in data["annotations"]] == [1, 2, 3, 4]
        assert [ann["image_id"] for ann in data["annotations"]] == [0, 1, 2, 3]
        assert [image["original_filename"] for image in data["images"]] == ["000003.jpg", "000008.jpg"] * 2
        assert data["annotations"][0]["bbox"] == [1, 0, 1, 2]
        assert data["annotations"][0]["depth_scale"] == 0.1
        assert data["annotations"][0]["full_segmentation"] != data["annotations"][0]["segmentation"]
        for image in data["images"]:
            image_path = Path(summary["image_dir"]) / image["file_name"]
            assert image_path.suffix == ".jpg" and not image_path.is_symlink()
            with Image.open(image_path) as decoded:
                assert decoded.format == "JPEG"
            assert (Path(summary["depth_dir"]) / image_path.with_suffix(".png").name).is_file()
        data["info"].pop("date_created")
        outputs.append(data)
    assert outputs[0] == outputs[1]


def test_failure_removes_partial_output_and_refuses_existing_directory(tmp_path):
    scene = _write_scene(tmp_path, "lmo", "test", 2)
    (scene / "mask/000008_000000.png").unlink()
    output = tmp_path / "failed"
    with pytest.raises(FileNotFoundError, match="mask/000008"):
        prepare_dataset(datasets_path=tmp_path, use_all_gt=True, output_dir=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))
    output.mkdir()
    sentinel = output / "existing.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_dataset(datasets_path=tmp_path, use_all_gt=True, output_dir=output)
    assert sentinel.read_text() == "keep"


def test_filters_limit_and_missing_inputs(tmp_path):
    _write_scene(tmp_path, "lmo", "test", 2)
    summary = prepare_dataset(datasets_path=tmp_path, use_all_gt=True, output_dir=tmp_path / "filtered",
                              scene_ids=[2], image_ids=[13, 8], max_images=1)
    assert [image["bop_im_id"] for image in _load_output(summary)["images"]] == [8]
    with pytest.raises(FileNotFoundError, match="test_targets_bop19"):
        prepare_dataset(datasets_path=tmp_path, output_dir=tmp_path / "no_targets")
    with pytest.raises(FileNotFoundError, match="train_pbr"):
        prepare_dataset(datasets_path=tmp_path, dataset_split="train", output_dir=tmp_path / "no_train")


@pytest.mark.parametrize("dataset,split", [("tless", "test_primesense"), ("hb", "test_primesense"), ("itodd", "test")])
def test_core_sensor_paths_and_grayscale_depth(tmp_path, dataset, split):
    gray = dataset == "itodd"
    _write_scene(tmp_path, dataset, split, 1, image_ids=(3,), extension=".tif" if gray else ".png", gray=gray)
    summary = prepare_dataset(dataset=dataset, datasets_path=tmp_path, use_all_gt=True,
                              output_dir=tmp_path / "out", include_depth=True)
    assert summary["images"] == 1
    assert _load_output(summary)["images"][0]["original_filename"] == ("000003.tif" if gray else "000003.png")
    with Image.open(Path(summary["depth_dir"]) / "000000.png") as depth:
        assert depth.format == "PNG"
        np.testing.assert_array_equal(np.asarray(depth), np.arange(12).reshape(3, 4) * 100)
