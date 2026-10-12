"""Headless, strict port of RACE6D/sub/test_kpt.ipynb (no notebook dependency).

The minimum-area XY box, 32-point construction, upper/lower ordering, manual
edge presets and symmetry-frame selection come from the original notebook.
Inputs are already in the training frame, including their models_info.
Do not regenerate a released checkpoint's cache and assume historical parity.
"""

import copy
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist

NOTEBOOK_PRESETS = {
    "ycbv": {
        "SPLIT_CONFIG": {
            "1": {"type": "z_zero"},
            "2": {"type": "z_zero"},
            "3": {"type": "z_zero"},
            "4": {"type": "z_zero"},
            "5": {"type": "z_zero"},
            "6": {"type": "z_zero"},
            "7": {"type": "z_zero"},
            "8": {"type": "z_zero"},
            "9": {"type": "z_zero"},
            "10": {"type": "z_zero"},
            "11": {"type": "z_zero"},
            "12": {"type": "z_zero"},
            "13": {"type": "z_zero"},
            "14": {"type": "z_zero"},
            "15": {"type": "custom_line", "point1": [60, 0], "point2": [-60, 0]},
            "16": {"type": "z_zero"},
            "17": {"type": "z_zero"},
            "18": {"type": "custom_line", "point1": [60, -10], "point2": [-60, 30]},
            "19": {"type": "z_zero"},
            "20": {"type": "z_zero"},
            "21": {"type": "z_zero"},
        },
        "EDGE_OVERRIDE": {
            "15": {
                "upper_edges": [[0, 1], [2, 3], [0, 2], [1, 3]],
                "lower_edges": [[0, 1], [2, 3], [0, 2], [1, 3]],
                "vertical_edges": [[0, 1], [1, 0], [2, 3], [3, 2]],
            },
            "18": {
                "upper_edges": [[0, 2], [0, 1], [1, 3], [2, 3]],
                "lower_edges": [[0, 2], [0, 1], [1, 3], [2, 3]],
                "vertical_edges": [[0, 1], [1, 0], [2, 3], [3, 2]],
            },
        },
    },
    "lmo": {
        "SPLIT_CONFIG": {
            "1": {"type": "z_zero"},
            "2": {"type": "z_zero"},
            "3": {"type": "z_zero"},
            "4": {"type": "z_zero"},
            "5": {"type": "z_zero"},
            "6": {"type": "z_zero"},
            "7": {"type": "z_zero"},
            "8": {"type": "z_zero"},
            "9": {"type": "z_zero"},
            "10": {"type": "z_zero"},
            "11": {"type": "z_zero"},
            "12": {"type": "z_zero"},
            "13": {"type": "z_zero"},
            "14": {"type": "z_zero"},
            "15": {"type": "z_zero"},
        },
        "EDGE_OVERRIDE": {
            "15": {
                "upper_edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "lower_edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "vertical_edges": [[0, 0], [1, 1], [2, 2], [3, 3]],
            },
        },
    },
    "hb": {
        "SPLIT_CONFIG": {
            "1": {"type": "z_zero"},
            "2": {"type": "z_zero"},
            "3": {"type": "z_zero"},
            "4": {"type": "z_zero"},
            "5": {"type": "z_zero"},
            "6": {"type": "z_zero"},
            "7": {"type": "z_zero"},
            "8": {"type": "z_zero"},
            "9": {"type": "z_zero"},
            "10": {"type": "z_zero"},
            "11": {"type": "z_zero"},
            "12": {"type": "z_zero"},
            "13": {"type": "z_zero"},
            "14": {"type": "z_zero"},
            "15": {"type": "z_zero"},
            "16": {"type": "z_zero"},
            "17": {"type": "z_zero"},
            "18": {"type": "z_zero"},
            "19": {"type": "z_zero"},
            "20": {"type": "z_zero"},
            "21": {"type": "z_zero"},
            "22": {"type": "z_zero"},
            "23": {"type": "z_zero"},
            "24": {"type": "z_zero"},
            "25": {"type": "z_zero"},
            "26": {"type": "z_zero"},
            "27": {"type": "z_zero"},
            "28": {"type": "z_zero"},
            "29": {"type": "z_zero"},
            "30": {"type": "z_zero"},
            "31": {"type": "z_zero"},
            "32": {"type": "z_zero"},
            "33": {"type": "z_zero"},
        },
        "EDGE_OVERRIDE": {
            "15": {
                "upper_edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "lower_edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "vertical_edges": [[0, 0], [1, 1], [2, 2], [3, 3]],
            },
            "18": {
                "upper_edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "lower_edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "vertical_edges": [[0, 0], [1, 1], [2, 2], [3, 3]],
            },
        },
    },
    "tudl": {
        "SPLIT_CONFIG": {
            "1": {"type": "z_zero"},
            "2": {"type": "z_zero"},
            "3": {"type": "z_zero"},
        },
        "EDGE_OVERRIDE": {},
    },
    "icbin": {
        "SPLIT_CONFIG": {
            "1": {"type": "z_zero"},
            "2": {"type": "z_zero"},
        },
        "EDGE_OVERRIDE": {},
    },
    "itodd": {
        "SPLIT_CONFIG": {
            "1": {"type": "z_zero"},
            "2": {"type": "z_zero"},
            "3": {"type": "z_zero"},
            "4": {"type": "z_zero"},
            "5": {"type": "z_zero"},
            "6": {"type": "z_zero"},
            "7": {"type": "z_zero"},
            "8": {"type": "z_zero"},
            "9": {"type": "z_zero"},
            "10": {"type": "z_zero"},
            "11": {"type": "z_zero"},
            "12": {"type": "z_zero"},
            "13": {"type": "z_zero"},
            "14": {"type": "z_zero"},
            "15": {"type": "z_zero"},
            "16": {"type": "z_zero"},
            "17": {"type": "z_zero"},
            "18": {"type": "z_zero"},
            "19": {"type": "z_zero"},
            "20": {"type": "z_zero"},
            "21": {"type": "z_zero"},
            "22": {"type": "z_zero"},
            "23": {"type": "z_zero"},
            "24": {"type": "z_zero"},
            "25": {"type": "z_zero"},
            "26": {"type": "z_zero"},
            "27": {"type": "z_zero"},
            "28": {"type": "z_zero"},
        },
        "EDGE_OVERRIDE": {},
    },
    "tless": {
        "SPLIT_CONFIG": {
            "1": {"type": "z_zero"},
            "2": {"type": "z_zero"},
            "3": {"type": "z_zero"},
            "4": {"type": "z_zero"},
            "5": {"type": "z_zero"},
            "6": {"type": "z_zero"},
            "7": {"type": "z_zero"},
            "8": {"type": "z_zero"},
            "9": {"type": "z_zero"},
            "10": {"type": "z_zero"},
            "11": {"type": "z_zero"},
            "12": {"type": "z_zero"},
            "13": {"type": "z_zero"},
            "14": {"type": "z_zero"},
            "15": {"type": "z_zero"},
            "16": {"type": "z_zero"},
            "17": {"type": "z_zero"},
            "18": {"type": "z_zero"},
            "19": {"type": "z_zero"},
            "20": {"type": "z_zero"},
            "21": {"type": "z_zero"},
            "22": {"type": "z_zero"},
            "23": {"type": "z_zero"},
            "24": {"type": "z_zero"},
            "25": {"type": "z_zero"},
            "26": {"type": "z_zero"},
            "27": {"type": "z_zero"},
            "28": {"type": "z_zero"},
            "29": {"type": "z_zero"},
            "30": {"type": "z_zero"},
        },
        "EDGE_OVERRIDE": {},
    },
}


def load_points_from_ply(path):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.has_triangles():
        o3d.utility.random.seed(0)
        cloud = mesh.sample_points_uniformly(number_of_points=5000)
    else:
        cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 3
        or not np.isfinite(points).all()
    ):
        raise ValueError("Invalid/empty PLY sample: {}".format(path))
    return points


def compute_2d_minimal_bbox(points):
    if len(points) < 3:
        return None

    xy_points = points[:, :2]
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    try:
        hull_2d = ConvexHull(xy_points)
        hull_points = xy_points[hull_2d.vertices]
    except:
        x_min, x_max = xy_points[:, 0].min(), xy_points[:, 0].max()
        y_min, y_max = xy_points[:, 1].min(), xy_points[:, 1].max()
        best_rect = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        )
    else:
        min_area = float("inf")
        best_rect = None
        angles = np.linspace(0, np.pi, 180)

        for angle in angles:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_hull = hull_points @ R.T
            x_min, x_max = rotated_hull[:, 0].min(), rotated_hull[:, 0].max()
            y_min, y_max = rotated_hull[:, 1].min(), rotated_hull[:, 1].max()
            area = (x_max - x_min) * (y_max - y_min)

            if area < min_area:
                min_area = area
                rect_corners_rot = np.array(
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                )
                best_rect = rect_corners_rot @ R

    corners_3d = np.array(
        [
            [best_rect[0][0], best_rect[0][1], z_min],
            [best_rect[1][0], best_rect[1][1], z_min],
            [best_rect[2][0], best_rect[2][1], z_min],
            [best_rect[3][0], best_rect[3][1], z_min],
            [best_rect[0][0], best_rect[0][1], z_max],
            [best_rect[1][0], best_rect[1][1], z_max],
            [best_rect[2][0], best_rect[2][1], z_max],
            [best_rect[3][0], best_rect[3][1], z_max],
        ],
        dtype=np.float64,
    )

    return corners_3d


def compute_aligned_bbox(points):
    """8 corners of the box aligned with the x, y, z axes (outermost points)."""
    lo, hi = points.min(0), points.max(0)
    rect = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    return np.array(
        [[x, y, z] for z in (lo[2], hi[2]) for x, y in rect], dtype=np.float64
    )


def generate_keypoints_32(corners):
    """32개 키포인트 생성: 0-7 corners, 8-31 edge midpoints"""
    keypoints = list(corners)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # vertical
    ]

    for i0, i1 in edges:
        A, B = corners[i0], corners[i1]
        keypoints.append((2 * A + 1 * B) / 3)
        keypoints.append((1 * A + 2 * B) / 3)

    return np.array(keypoints, dtype=np.float64)


# ==================== 4. Upper/Lower 분할 ====================
def split_by_z_zero(keypoints):
    upper_mask = keypoints[:, 2] > 0
    lower_mask = keypoints[:, 2] <= 0
    return upper_mask, lower_mask


def split_by_custom_line(keypoints, point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1

    xy_coords = keypoints[:, :2]
    distances = a * xy_coords[:, 0] + b * xy_coords[:, 1] + c
    upper_mask = distances > 0
    lower_mask = distances <= 0
    return upper_mask, lower_mask


# ==================== 5. Corners 추적 재배열 ====================
def rearrange_with_corners(keypoints_32, mask):
    corner_mask = np.zeros(32, dtype=bool)
    corner_mask[:8] = True

    selected_corners_mask = mask & corner_mask
    selected_non_corners_mask = mask & ~corner_mask

    corners = keypoints_32[selected_corners_mask]
    non_corners = keypoints_32[selected_non_corners_mask]

    n_corners = len(corners)
    n_remaining = 16 - n_corners

    if n_remaining <= 0:
        result = corners[:16]
        corner_indices = list(range(min(n_corners, 16)))
    else:
        if len(non_corners) >= n_remaining:
            indices = np.linspace(0, len(non_corners) - 1, n_remaining, dtype=int)
            selected_non = non_corners[indices]
        elif len(non_corners) > 0:
            padding = np.zeros((n_remaining - len(non_corners), 3))
            selected_non = np.vstack([non_corners, padding])
        else:
            selected_non = np.zeros((n_remaining, 3))

        result = np.vstack([corners, selected_non])
        corner_indices = list(range(n_corners))

    return result, corner_indices


# ==================== 6. Cross-ratio 계산 ====================
def compute_cross_ratio(points):
    """4개 점의 cross-ratio: CR(A,B,C,D) = (AC*BD)/(AD*BC)"""
    if len(points) != 4:
        return None

    A, B, C, D = points

    AC = np.linalg.norm(C - A)
    BD = np.linalg.norm(D - B)
    AD = np.linalg.norm(D - A)
    BC = np.linalg.norm(C - B)

    if AD < 1e-10 or BC < 1e-10:
        return None

    cr = (AC * BD) / (AD * BC)
    return cr


# ==================== 7. Edges Mask 재매핑 (Cross-ratio) ====================
def find_corner_edges(corner_points, corner_indices):
    """거리 기반으로 corners 간의 edges 찾기"""
    n = len(corner_points)
    if n < 2:
        return []

    distances = cdist(corner_points, corner_points)
    edges = []
    used = set()

    for i in range(n):
        dists = [(j, distances[i][j]) for j in range(n) if j != i]
        dists.sort(key=lambda x: x[1])

        for j, d in dists[:2]:
            edge = tuple(sorted([corner_indices[i], corner_indices[j]]))
            if edge not in used:
                edges.append((corner_indices[i], corner_indices[j]))
                used.add(edge)
                if len(edges) >= 4:
                    return edges[:4]

    return edges[:4]


def find_edge_midpoints_cross_ratio(total_kpts, c1, c2, used_points):
    """Cross-ratio 체크로 완벽한 중간점 찾기"""
    p1, p2 = total_kpts[c1], total_kpts[c2]
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-6:
        return [c1, c1, c2, c2]

    # 직선에 가까운 후보점들
    candidates = []
    for i in range(len(total_kpts)):
        if i in used_points or i == c1 or i == c2:
            continue

        point = total_kpts[i]
        vec = point - p1
        proj_len = np.dot(vec, line_vec) / line_len

        if 0 <= proj_len <= line_len:
            proj_point = p1 + (proj_len / line_len) * line_vec
            dist = np.linalg.norm(point - proj_point)
            candidates.append((i, dist, proj_len))

    if len(candidates) < 2:
        # 후보 부족 - fallback
        candidates.sort(key=lambda x: x[1])
        mids = [x[0] for x in candidates] if candidates else []
        all_pts = [c1] + mids + [c2]
        while len(all_pts) < 4:
            all_pts.insert(-1, c2)
        return all_pts[:4]

    # Cross-ratio 기반 최적 조합 찾기
    target_cr = 4.0 / 3.0
    best_combo = None
    best_error = float("inf")

    candidates.sort(key=lambda x: x[1])
    top_candidates = candidates[: min(15, len(candidates))]

    # 2개 중간점 조합 시도
    for combo in combinations(top_candidates, 2):
        mid_indices = [x[0] for x in combo]
        all_pts = [c1] + mid_indices + [c2]

        # 위치순 정렬
        all_pts.sort(
            key=lambda idx: (
                np.dot(total_kpts[idx] - p1, line_vec) / line_len if line_len > 0 else 0
            )
        )

        if len(all_pts) == 4:
            points_3d = total_kpts[all_pts]
            cr = compute_cross_ratio(points_3d)

            if cr is not None:
                error = abs(cr - target_cr)

                if error < best_error:
                    best_error = error
                    best_combo = all_pts

    # 좋은 조합 발견 (5% 오차 이내)
    if best_combo is not None and best_error < 0.067:  # 4/3 * 0.05
        return best_combo

    # Fallback: 거리순
    candidates.sort(key=lambda x: x[1])
    mids = [x[0] for x in candidates[:2]]
    all_pts = [c1] + mids + [c2]
    all_pts.sort(
        key=lambda idx: (
            np.dot(total_kpts[idx] - p1, line_vec) / line_len if line_len > 0 else 0
        )
    )

    while len(all_pts) < 4:
        all_pts.insert(-1, c2)

    return all_pts[:4]


def create_manual_edges_mask(
    upper_kpts, lower_kpts, upper_corners, lower_corners, edge_config
):
    total_kpts = np.vstack([upper_kpts, lower_kpts])
    n_upper = len(upper_kpts)

    edges_mask = {}
    used_points = set()

    # Upper edges
    if "upper_edges" in edge_config:
        for i, (c1_idx, c2_idx) in enumerate(edge_config["upper_edges"]):
            if c1_idx < len(upper_corners) and c2_idx < len(upper_corners):
                c1 = upper_corners[c1_idx]
                c2 = upper_corners[c2_idx]
                edge_pts = find_edge_midpoints_cross_ratio(
                    total_kpts, c1, c2, used_points
                )
                if len(edge_pts) == 4:
                    edges_mask[f"upper_edge_{i+1}"] = edge_pts
                    for pt in edge_pts[1:-1]:
                        used_points.add(pt)

    # Lower edges
    lower_corners_adj = [idx + n_upper for idx in lower_corners]
    if "lower_edges" in edge_config:
        for i, (c1_idx, c2_idx) in enumerate(edge_config["lower_edges"]):
            if c1_idx < len(lower_corners) and c2_idx < len(lower_corners):
                c1 = lower_corners_adj[c1_idx]
                c2 = lower_corners_adj[c2_idx]
                edge_pts = find_edge_midpoints_cross_ratio(
                    total_kpts, c1, c2, used_points
                )
                if len(edge_pts) == 4:
                    edges_mask[f"lower_edge_{i+1}"] = edge_pts
                    for pt in edge_pts[1:-1]:
                        used_points.add(pt)

    # Vertical edges
    if "vertical_edges" in edge_config:
        for i, (u_idx, l_idx) in enumerate(edge_config["vertical_edges"]):
            if u_idx < len(upper_corners) and l_idx < len(lower_corners):
                c1 = upper_corners[u_idx]
                c2 = lower_corners_adj[l_idx]
                edge_pts = find_edge_midpoints_cross_ratio(
                    total_kpts, c1, c2, used_points
                )
                if len(edge_pts) == 4:
                    edges_mask[f"vertical_edge_{i+1}"] = edge_pts
                    for pt in edge_pts[1:-1]:
                        used_points.add(pt)

    return edges_mask


def create_edges_mask_cross_ratio(upper_kpts, lower_kpts, upper_corners, lower_corners):
    """Cross-ratio 검증으로 Edges mask 생성"""
    total_kpts = np.vstack([upper_kpts, lower_kpts])
    n_upper = len(upper_kpts)

    lower_corners_adj = [idx + n_upper for idx in lower_corners]

    edges_mask = {}
    used_points = set()

    # Upper edges
    if len(upper_corners) >= 2:
        upper_corner_pts = upper_kpts[upper_corners]
        upper_edge_pairs = find_corner_edges(upper_corner_pts, upper_corners)

        for i, (c1, c2) in enumerate(upper_edge_pairs):
            edge_pts = find_edge_midpoints_cross_ratio(total_kpts, c1, c2, used_points)
            if len(edge_pts) == 4:
                edges_mask[f"upper_edge_{i+1}"] = edge_pts
                for pt in edge_pts[1:-1]:
                    used_points.add(pt)

    # Lower edges
    if len(lower_corners) >= 2:
        lower_corner_pts = lower_kpts[lower_corners]
        lower_edge_pairs = find_corner_edges(lower_corner_pts, lower_corners_adj)

        for i, (c1, c2) in enumerate(lower_edge_pairs):
            edge_pts = find_edge_midpoints_cross_ratio(total_kpts, c1, c2, used_points)
            if len(edge_pts) == 4:
                edges_mask[f"lower_edge_{i+1}"] = edge_pts
                for pt in edge_pts[1:-1]:
                    used_points.add(pt)

    # Vertical edges
    n_vert = min(len(upper_corners), len(lower_corners))
    for i in range(n_vert):
        c1 = upper_corners[i]
        c2 = lower_corners_adj[i]
        edge_pts = find_edge_midpoints_cross_ratio(total_kpts, c1, c2, used_points)
        if len(edge_pts) == 4:
            edges_mask[f"vertical_edge_{i+1}"] = edge_pts
            for pt in edge_pts[1:-1]:
                used_points.add(pt)

    return edges_mask


# ==================== 8. 파이프라인 ====================
def _rotation_axis_angle(R):
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
    ev, evec = np.linalg.eig(R)
    axis = np.real(evec[:, np.argmin(np.abs(ev - 1))])
    return axis / np.linalg.norm(axis), angle


def _axes_frame(z, x):
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def _process_points(points, config, edge_config=None, aligned=False):
    corners = (
        compute_2d_minimal_bbox(points) if not aligned else compute_aligned_bbox(points)
    )
    if corners is None:
        return None
    keypoints_32 = generate_keypoints_32(corners)

    if config["type"] == "z_zero":
        upper_mask, lower_mask = split_by_z_zero(keypoints_32)
    else:
        upper_mask, lower_mask = split_by_custom_line(
            keypoints_32, config["point1"], config["point2"]
        )

    upper_16, upper_corners = rearrange_with_corners(keypoints_32, upper_mask)
    lower_16, lower_corners = rearrange_with_corners(keypoints_32, lower_mask)
    keypoints_final = np.vstack([upper_16, lower_16])

    # Edges mask 생성 (override 체크)
    if edge_config:
        edges_mask = create_manual_edges_mask(
            upper_16, lower_16, upper_corners, lower_corners, edge_config
        )
    else:
        edges_mask = create_edges_mask_cross_ratio(
            upper_16, lower_16, upper_corners, lower_corners
        )

    # Cross-ratio 검증
    cr_errors = []
    for edge_name, edge_indices in edges_mask.items():
        if len(edge_indices) == 4:
            points_4 = keypoints_final[edge_indices]
            cr = compute_cross_ratio(points_4)
            if cr is not None:
                error = abs(cr - 4.0 / 3.0)
                cr_errors.append(error)

    avg_cr_error = np.mean(cr_errors) if cr_errors else 0.0

    return {
        "ply_points": points,
        "keypoints": keypoints_final,
        "upper_kpts": upper_16,
        "lower_kpts": lower_16,
        "edges_mask": edges_mask,
        "upper_corners": upper_corners,
        "lower_corners": lower_corners,
        "method": config["type"],
        "avg_cr_error": avg_cr_error,
    }


def _notebook_symmetry_axis(info):
    """Use the notebook's axis choice, not the all-constraint P audit.

    test_kpt.ipynb imports the legacy compute_symmetry_centers.symmetry_axis:
    first continuous axis takes precedence; otherwise solve discrete entries.
    Combining both changes the box frame for ITODD 12/28. The independent
    model-origin preparation still audits all supplied symmetry constraints.
    """
    if info.get("symmetries_continuous"):
        symmetry = info["symmetries_continuous"][0]
        axis = np.asarray(symmetry["axis"], dtype=np.float64)
        return (
            np.asarray(symmetry["offset"], dtype=np.float64),
            axis / np.linalg.norm(axis),
            0.0,
        )
    if info.get("symmetries_discrete"):
        matrices = [
            np.asarray(value, dtype=np.float64).reshape(4, 4)
            for value in info["symmetries_discrete"]
        ]
        equation = np.concatenate([np.eye(3) - m[:3, :3] for m in matrices])
        value = np.concatenate([m[:3, 3] for m in matrices])
        point = np.linalg.lstsq(equation, value, rcond=None)[0]
        _, singular, right = np.linalg.svd(equation)
        axis = right[-1] if singular[-1] < 1e-6 * singular[0] else None
        return point, axis, float(np.abs(equation @ point - value).max())
    return None


def symmetry_axis_frame(info, points):
    """Reproduce the notebook's box-frame policy in the mesh's current frame.

    Here models_info has already followed v-P. The notebook instead adjusts
    its original metadata with T_out @ symmetry @ T_in and point-P. These
    are the same coordinate change; never subtract P a second time here.
    """
    found = _notebook_symmetry_axis(info)
    if found is None:
        return None
    point, axis, _ = found
    if info.get("symmetries_continuous"):
        transforms = []
        # The notebook tests 90/180/270 degrees around its chosen continuous
        # axis only; discrete flips are NOT added to this box-selection test.
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        for degree in (90, 180, 270):
            angle = np.deg2rad(degree)
            matrix = np.eye(4)
            matrix[:3, :3] = (
                np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew
            )
            matrix[:3, 3] = point - matrix[:3, :3] @ point
            transforms.append(matrix)
    else:
        transforms = [
            np.asarray(m, dtype=np.float64).reshape(4, 4)
            for m in info["symmetries_discrete"]
        ]
    if axis is not None:
        index = int(np.argmax(np.abs(axis)))
        if axis[index] < 0:
            axis = -axis
        point = point - axis * (point @ axis)
        return (
            point,
            _axes_frame(axis, np.eye(3)[0] if index != 0 else np.eye(3)[1]),
            transforms,
            False,
        )
    rotations = [_rotation_axis_angle(m[:3, :3]) for m in transforms]
    smallest = min(angle for _, angle in rotations)
    z_candidates = [a for a, angle in rotations if abs(angle - smallest) < 1]
    z = max(z_candidates, key=lambda a: abs(a[2]))
    z *= np.sign(z[np.argmax(np.abs(z))])
    x_candidates = [
        a for a, angle in rotations if abs(angle - 180) < 1 and abs(a @ z) < 0.1
    ]
    if not x_candidates:
        return None, None, transforms, False

    def rectangle_area(x):
        local = (points - point) @ _axes_frame(z, x)
        return np.prod(local[:, :2].max(0) - local[:, :2].min(0))

    return (
        point,
        _axes_frame(z, min(x_candidates, key=rectangle_area)),
        transforms,
        True,
    )


def keypoint_symmetry_error(keypoints, transforms):
    if not transforms:
        return 0.0
    tree = cKDTree(keypoints)
    return float(
        max(tree.query(keypoints @ m[:3, :3].T + m[:3, 3])[0].max() for m in transforms)
    )


def validate_keypoints(keypoints, edges):
    keypoints = np.asarray(keypoints, dtype=np.float64)
    edges = np.asarray(edges)
    if keypoints.shape != (32, 3) or not np.isfinite(keypoints).all():
        raise ValueError("Expected 32 finite 3D keypoints")
    if np.unique(np.round(keypoints, 10), axis=0).shape[0] != 32:
        raise ValueError(
            "Split/padding created duplicate keypoints; provide a suitable split_config"
        )
    if (
        edges.shape != (12, 4)
        or not np.issubdtype(edges.dtype, np.integer)
        or np.any(edges < 0)
        or np.any(edges >= 32)
    ):
        raise ValueError("Expected twelve integer 4-point edges with indices in [0,31]")
    if (
        np.unique(edges).size != 32
        or len({tuple(sorted((int(edge[0]), int(edge[-1])))) for edge in edges}) != 12
    ):
        raise ValueError(
            "Edges must cover all 32 keypoints and describe twelve different box edges"
        )
    errors = []
    for indices in edges:
        if len(set(indices.tolist())) != 4:
            raise ValueError("An edge repeats a keypoint")
        a, b, c, d = keypoints[indices]
        vector = d - a
        length = np.linalg.norm(vector)
        if length < 1e-9:
            raise ValueError("Zero-length edge")
        projected = (keypoints[indices] - a) @ vector / length**2
        off_line = keypoints[indices] - a - projected[:, None] * vector
        if np.max(np.linalg.norm(off_line, axis=1)) > max(
            1e-6, 1e-5 * length
        ) or not np.allclose(projected, [0, 1 / 3, 2 / 3, 1], atol=1e-5, rtol=0):
            raise ValueError(
                "Edge is not collinear and evenly spaced; check edge_override"
            )
        error = abs(compute_cross_ratio(keypoints[indices]) - 4 / 3)
        errors.append(error)
    return float(max(errors))


def generate_keypoints(points, info, split_config, edge_config=None, axis_box=True):
    # Keep the same sampled-point dtype as the notebook. A forced float64
    # conversion changes the base XY/Z calculation for float32 input clouds.
    points = np.asarray(points)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 3
        or not np.isfinite(points).all()
    ):
        raise ValueError("Expected finite [N,3] model points")
    if split_config.get("type") not in ("z_zero", "custom_line"):
        raise ValueError("Split type must be z_zero or custom_line")
    if split_config["type"] == "custom_line":
        line = np.asarray([split_config["point1"], split_config["point2"]], dtype=float)
        if (
            line.shape != (2, 2)
            or not np.isfinite(line).all()
            or np.linalg.norm(line[0] - line[1]) < 1e-9
        ):
            raise ValueError("Invalid custom split line")
    if edge_config:
        allowed = {"upper_edges", "lower_edges", "vertical_edges"}
        if not isinstance(edge_config, dict) or set(edge_config) != allowed:
            raise ValueError("An edge override must define upper/lower/vertical_edges")
        for pairs in edge_config.values():
            pairs = np.asarray(pairs)
            if (
                pairs.shape != (4, 2)
                or not np.issubdtype(pairs.dtype, np.integer)
                or np.any(pairs < 0)
                or np.any(pairs >= 4)
            ):
                raise ValueError(
                    "Edge override needs four integer corner pairs in [0,3]"
                )
    result = _process_points(points, split_config, edge_config)
    if result is None:
        raise ValueError("Could not construct a bounding box")
    symmetry = symmetry_axis_frame(info, points)
    transforms = symmetry[2] if symmetry else []
    result["box_frame"] = "model"
    result["sym_error"] = keypoint_symmetry_error(result["keypoints"], transforms)
    if axis_box and symmetry and symmetry[0] is not None:
        point, frame, _, aligned = symmetry
        other = _process_points(
            (points - point) @ frame, split_config, edge_config, aligned
        )
        if (
            other is not None
            and len(other["edges_mask"]) == 12
            and other["avg_cr_error"] < 1e-3
            and all(len(edge) == 4 for edge in other["edges_mask"].values())
        ):
            for name in ("ply_points", "keypoints", "upper_kpts", "lower_kpts"):
                other[name] = other[name] @ frame.T + point
            other["sym_error"] = keypoint_symmetry_error(other["keypoints"], transforms)
            if other["sym_error"] < result["sym_error"] - 0.1:
                other["box_frame"] = "axis"
                result = other
    result["max_cr_error"] = validate_keypoints(
        result["keypoints"],
        [result["edges_mask"][name] for name in sorted(result["edges_mask"])],
    )
    return result


def generate_cache(
    models_path, object_ids, dataset, keypoint_config=None, axis_box=True
):
    models_path = Path(models_path)
    info = json.loads((models_path / "models_info.json").read_text())
    preset = copy.deepcopy(
        NOTEBOOK_PRESETS.get(dataset, {"SPLIT_CONFIG": {}, "EDGE_OVERRIDE": {}})
    )
    if keypoint_config is not None:
        if not isinstance(keypoint_config, dict) or set(keypoint_config) - {
            "split_config",
            "edge_override",
        }:
            raise ValueError(
                "keypoint_config accepts only split_config and edge_override mappings"
            )
        for name, target in (
            ("split_config", "SPLIT_CONFIG"),
            ("edge_override", "EDGE_OVERRIDE"),
        ):
            if name in keypoint_config:
                if not isinstance(keypoint_config[name], dict):
                    raise ValueError(name + " must be an object-ID mapping")
                preset[target].update(keypoint_config[name])
    # JSON presets/configs use strings, while notebook constants originally used ints.
    splits = {int(key): value for key, value in preset["SPLIT_CONFIG"].items()}
    overrides = {int(key): value for key, value in preset["EDGE_OVERRIDE"].items()}
    cache = {
        "metadata": {
            "description": "RACE6D 32-point training-frame box keypoints",
            "generation_method": "Headless notebook port: 180-angle XY box, Z bounds, optional symmetry box",
            "coordinate_system": "training_model_frame",
            "units": "mm",
            "num_keypoints_per_class": 32,
            "edges_per_object": 12,
            "cross_ratio": "4/3",
            "sampling": "Open3D surface 5000, seed 0; vertices for point-only PLY",
            "symmetry_frame_policy": "notebook: first continuous axis, otherwise discrete; no mixed-axis redefinition",
            "generated_classes": list(object_ids),
            "object_audit": {},
        },
        "keypoints_3d": {},
        "edges_mask": {},
    }
    for obj_id in object_ids:
        key = str(obj_id)
        try:
            result = generate_keypoints(
                load_points_from_ply(models_path / "obj_{:06d}.ply".format(obj_id)),
                info[key],
                splits.get(obj_id, {"type": "z_zero"}),
                overrides.get(obj_id),
                axis_box,
            )
        except (ValueError, KeyError) as error:
            raise ValueError("obj {}: {}".format(obj_id, error)) from error
        cache["keypoints_3d"][key] = result["keypoints"].tolist()
        cache["edges_mask"][key] = [
            list(result["edges_mask"][name]) for name in sorted(result["edges_mask"])
        ]
        cache["metadata"]["object_audit"][key] = {
            "box_frame": result["box_frame"],
            "max_cross_ratio_error": result["max_cr_error"],
            "symmetry_error_mm": result["sym_error"],
        }
    return cache
