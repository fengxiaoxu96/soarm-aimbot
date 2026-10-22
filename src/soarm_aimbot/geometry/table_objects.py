"""Single-frame table objects from an overhead depth map.

The overhead depth is backprojected with the step-1 intrinsics and mapped
into the table frame with the step-3 transform. A plane is fit to the table,
points in a height band are clustered, and each cluster becomes one object.
Stable ids across frames are kept by ``geometry.tracks``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics
from soarm_aimbot.geometry.transforms import RigidTransform

IgnoreBox = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class TablePlane:
    """Plane ``normal · point = offset`` in the world frame. The normal points upward."""

    normal: np.ndarray
    offset_m: float
    inlier_count: int
    inlier_fraction: float

    def height(self, points: np.ndarray) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        return array @ self.normal - self.offset_m


@dataclass(frozen=True)
class TableObject:
    """One cluster in a single frame. ``temporary_id`` is not stable across frames."""

    temporary_id: int
    position_m: np.ndarray
    extent_m: np.ndarray
    bbox_min_m: np.ndarray
    bbox_max_m: np.ndarray
    height_above_table_m: float
    point_count: int
    pixels: np.ndarray


@dataclass(frozen=True)
class Segmentation:
    plane: TablePlane
    objects: list[TableObject]
    kept_points: np.ndarray
    kept_labels: np.ndarray
    plane_points: np.ndarray


def segment_depth(
    depth_m: np.ndarray,
    intrinsics: PinholeIntrinsics,
    world_from_overhead: RigidTransform,
    *,
    pixel_keep: np.ndarray | None = None,
    ignore_rois: list[tuple[int, int, int, int]] | None = None,
    min_height_m: float = 0.005,
    max_height_m: float = 0.300,
    cluster_m: float = 0.015,
    min_points: int = 80,
    percentile: float = 2.0,
    plane_distance_m: float = 0.008,
    iterations: int = 200,
    seed: int = 0,
    xy_min: tuple[float, float] | None = None,
    xy_max: tuple[float, float] | None = None,
    ignore_boxes: list[IgnoreBox] | None = None,
) -> Segmentation:
    """Backproject a depth map and segment objects in the table frame.

    ``pixel_keep`` is an optional boolean image. False pixels can still support
    the table plane, but they are not clustered into objects. ``ignore_rois``
    are removed before the plane fit.
    """

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"Depth map shape {depth.shape} does not match the intrinsics {intrinsics.width}x{intrinsics.height}"
        )
    valid = depth > 0
    for x, y, width, height in ignore_rois or []:
        _validate_roi(depth.shape, (x, y, width, height))
        valid[y : y + height, x : x + width] = False
    if pixel_keep is not None:
        mask = np.asarray(pixel_keep, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError(f"Pixel mask shape {mask.shape} does not match the depth map {depth.shape}")
        keep = mask[valid]
    else:
        keep = None
    rows, cols = np.nonzero(valid)
    if rows.size < 50:
        raise ValueError("Fewer than 50 valid depth points; cannot fit the table")
    camera_points = intrinsics.backproject(cols, rows, depth[rows, cols])
    world_points = world_from_overhead.apply(camera_points)
    pixels = np.stack([cols, rows], axis=1).astype(np.int32)
    return segment_world_points(
        world_points,
        pixels=pixels,
        keep=keep,
        min_height_m=min_height_m,
        max_height_m=max_height_m,
        cluster_m=cluster_m,
        min_points=min_points,
        percentile=percentile,
        plane_distance_m=plane_distance_m,
        iterations=iterations,
        seed=seed,
        xy_min=xy_min,
        xy_max=xy_max,
        ignore_boxes=ignore_boxes,
    )


def segment_world_points(
    points: np.ndarray,
    *,
    pixels: np.ndarray | None = None,
    keep: np.ndarray | None = None,
    min_height_m: float = 0.005,
    max_height_m: float = 0.300,
    cluster_m: float = 0.015,
    min_points: int = 80,
    percentile: float = 2.0,
    plane_distance_m: float = 0.008,
    iterations: int = 200,
    seed: int = 0,
    xy_min: tuple[float, float] | None = None,
    xy_max: tuple[float, float] | None = None,
    ignore_boxes: list[IgnoreBox] | None = None,
) -> Segmentation:
    """Segment points that are already in the world frame."""

    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"The point cloud must be (N, 3), got {cloud.shape}")
    if not 0.0 <= percentile < 50.0:
        raise ValueError(f"The percentile must be between 0 and 50, got {percentile}")
    if cluster_m <= 0 or min_height_m < 0 or max_height_m <= min_height_m:
        raise ValueError("Invalid height band or cluster distance")
    if pixels is not None and len(pixels) != len(cloud):
        raise ValueError("The number of pixel coordinates does not match the number of points")
    if keep is not None and len(keep) != len(cloud):
        raise ValueError("The mask length does not match the number of points")

    in_workspace = _xy_mask(cloud, xy_min, xy_max)
    blocked = _in_boxes(cloud, ignore_boxes or [])
    plane_points = cloud[in_workspace & ~blocked]
    plane = fit_table_plane(
        plane_points,
        distance_m=plane_distance_m,
        iterations=iterations,
        seed=seed,
    )
    height = plane.height(cloud)
    selected = (
        in_workspace
        & ~blocked
        & (height >= min_height_m)
        & (height <= max_height_m)
    )
    if keep is not None:
        selected &= np.asarray(keep, dtype=bool)
    selected_index = np.flatnonzero(selected)
    clusters = cluster_indices(cloud[selected], cluster_m, min_points)
    objects: list[tuple[np.ndarray, TableObject]] = []
    for local_index in clusters:
        original = selected_index[local_index]
        sample = cloud[original]
        low = np.percentile(sample, percentile, axis=0)
        high = np.percentile(sample, 100.0 - percentile, axis=0)
        if pixels is None:
            object_pixels = np.zeros((0, 2), dtype=np.int32)
        else:
            object_pixels = np.asarray(pixels, dtype=np.int32)[original]
        objects.append(
            (
                original,
                TableObject(
                    temporary_id=-1,
                    position_m=np.median(sample, axis=0),
                    extent_m=high - low,
                    bbox_min_m=low,
                    bbox_max_m=high,
                    height_above_table_m=float(np.median(height[original])),
                    point_count=int(len(original)),
                    pixels=object_pixels,
                ),
            )
        )
    objects.sort(key=lambda item: (float(item[1].position_m[0]), float(item[1].position_m[1])))
    labels = np.full(len(selected_index), -1, dtype=np.int32)
    slot_of = np.full(len(cloud), -1, dtype=np.int32)
    slot_of[selected_index] = np.arange(len(selected_index), dtype=np.int32)
    ordered: list[TableObject] = []
    for temporary_id, (original, item) in enumerate(objects):
        labels[slot_of[original]] = temporary_id
        ordered.append(
            TableObject(
                temporary_id=temporary_id,
                position_m=item.position_m,
                extent_m=item.extent_m,
                bbox_min_m=item.bbox_min_m,
                bbox_max_m=item.bbox_max_m,
                height_above_table_m=item.height_above_table_m,
                point_count=item.point_count,
                pixels=item.pixels,
            )
        )
    plane_draw = plane_points
    if len(plane_draw) > 8000:
        generator = np.random.default_rng(seed)
        plane_draw = plane_draw[generator.choice(len(plane_draw), 8000, replace=False)]
    return Segmentation(
        plane=plane,
        objects=ordered,
        kept_points=cloud[selected],
        kept_labels=labels,
        plane_points=plane_draw,
    )


def fit_table_plane(
    points: np.ndarray,
    *,
    distance_m: float,
    iterations: int = 200,
    seed: int = 0,
) -> TablePlane:
    """RANSAC plane. The normal is flipped so its world-Z component is positive."""

    cloud = np.asarray(points, dtype=np.float64)
    if len(cloud) < 50:
        raise ValueError(f"Only {len(cloud)} points for the plane fit; at least 50 are needed")
    if distance_m <= 0:
        raise ValueError(f"The plane distance threshold must be positive, got {distance_m}")
    generator = np.random.default_rng(seed)
    sample_count = min(len(cloud), 30000)
    sample = cloud[generator.choice(len(cloud), sample_count, replace=False)]
    best_count = -1
    best_normal = np.array([0.0, 0.0, 1.0])
    best_offset = 0.0
    for _ in range(iterations):
        chosen = sample[generator.choice(len(sample), 3, replace=False)]
        normal = np.cross(chosen[1] - chosen[0], chosen[2] - chosen[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            continue
        normal = normal / length
        if normal[2] < 0.0:
            normal = -normal
        offset = float(normal @ chosen[0])
        count = int(np.count_nonzero(np.abs(sample @ normal - offset) <= distance_m))
        if count > best_count:
            best_count = count
            best_normal = normal
            best_offset = offset
    if best_count < 50:
        raise ValueError("No large enough table plane was found")
    inliers = cloud[np.abs(cloud @ best_normal - best_offset) <= distance_m]
    if len(inliers) > 50000:
        inliers = inliers[generator.choice(len(inliers), 50000, replace=False)]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    if abs(float(normal[2])) < 0.7:
        raise ValueError(
            "The fitted plane is not level enough. Keep --xy inside the table so the floor or a vertical background does not dominate."
        )
    offset = float(normal @ centroid)
    inlier_count = int(np.count_nonzero(np.abs(cloud @ normal - offset) <= distance_m))
    return TablePlane(
        normal=normal,
        offset_m=offset,
        inlier_count=inlier_count,
        inlier_fraction=float(inlier_count / len(cloud)),
    )


def cluster_indices(points: np.ndarray, radius_m: float, min_points: int) -> list[np.ndarray]:
    """Euclidean clusters via 26-connected voxels of edge ``radius_m``.

    Points farther apart than about ``radius_m * sqrt(3)`` stay in different
    clusters. Clusters smaller than ``min_points`` are dropped.
    """

    cloud = np.asarray(points, dtype=np.float64)
    if len(cloud) == 0 or min_points <= 0:
        return []
    keys = np.floor(cloud / radius_m).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    if len(keys_sorted) == 1:
        boundaries = np.array([0, 1])
    else:
        change = np.any(keys_sorted[1:] != keys_sorted[:-1], axis=1)
        boundaries = np.concatenate([[0], np.flatnonzero(change) + 1, [len(keys_sorted)]])
    voxel_ids: list[tuple[int, int, int]] = []
    voxel_points: list[np.ndarray] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        voxel_ids.append(
            (int(keys_sorted[start, 0]), int(keys_sorted[start, 1]), int(keys_sorted[start, 2]))
        )
        voxel_points.append(order[start:end])
    parent = list(range(len(voxel_ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    lookup = {voxel_id: index for index, voxel_id in enumerate(voxel_ids)}
    for index, (x_index, y_index, z_index) in enumerate(voxel_ids):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    other = lookup.get((x_index + dx, y_index + dy, z_index + dz))
                    if other is not None and other > index:
                        union(index, other)
    grouped: dict[int, list[int]] = {}
    for index in range(len(voxel_ids)):
        grouped.setdefault(find(index), []).append(index)
    clusters: list[np.ndarray] = []
    for members in grouped.values():
        indices = np.concatenate([voxel_points[member] for member in members])
        if len(indices) >= min_points:
            clusters.append(indices)
    return clusters


def _xy_mask(
    points: np.ndarray,
    xy_min: tuple[float, float] | None,
    xy_max: tuple[float, float] | None,
) -> np.ndarray:
    mask = np.ones(len(points), dtype=bool)
    if xy_min is not None:
        mask &= (points[:, 0] >= xy_min[0]) & (points[:, 1] >= xy_min[1])
    if xy_max is not None:
        mask &= (points[:, 0] <= xy_max[0]) & (points[:, 1] <= xy_max[1])
    return mask


def _in_boxes(points: np.ndarray, boxes: list[IgnoreBox]) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for xmin, ymin, zmin, xmax, ymax, zmax in boxes:
        inside |= (
            (points[:, 0] >= min(xmin, xmax))
            & (points[:, 0] <= max(xmin, xmax))
            & (points[:, 1] >= min(ymin, ymax))
            & (points[:, 1] <= max(ymin, ymax))
            & (points[:, 2] >= min(zmin, zmax))
            & (points[:, 2] <= max(zmin, zmax))
        )
    return inside


def _validate_roi(shape: tuple[int, ...], roi: tuple[int, int, int, int]) -> None:
    x, y, width, height = roi
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > shape[1] or y + height > shape[0]:
        raise ValueError(f"Ignore box {roi} is outside the {shape[1]}x{shape[0]} image")
