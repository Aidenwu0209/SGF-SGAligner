"""CPU-only local RGB-D constraint experiments. No learned matcher or GT access.

Geometric overlap ROI is a proxy for region guidance, not semantic guidance.
Fit frames and held-out checking frames are disjoint. All profiles retain
independent bidirectional geometric checks; color is not an independent vote.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import time
import numpy as np

from .contracts import validate_se3, sha256_file


@dataclass(frozen=True)
class LightweightConfig:
    fit_offsets: tuple[int, ...] = (-8, 0, 8)
    check_offset: int = 4
    pixel_stride: int = 6
    maximum_points: int = 12000
    cloud_voxel_m: float = 0.025
    roi_distance_m: float = 0.15
    minimum_points: int = 200
    maximum_update_translation_m: float = 0.10
    maximum_update_rotation_deg: float = 5.0
    minimum_validation_improvement: float = 0.01
    minimum_coverage_retention: float = 0.95

    def __post_init__(self):
        if self.check_offset in self.fit_offsets:
            raise ValueError("fit and checking frame offsets overlap")
        if min(self.pixel_stride, self.maximum_points, self.minimum_points) < 1:
            raise ValueError("point budgets must be positive")
        for value in (self.cloud_voxel_m, self.roi_distance_m,
                      self.maximum_update_translation_m, self.maximum_update_rotation_deg):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("limits must be finite and positive")
        if not 0 <= self.minimum_validation_improvement < 1 or not 0 < self.minimum_coverage_retention <= 1:
            raise ValueError("invalid validation limits")


def frame_indices(anchor: int, count: int, config: LightweightConfig):
    fit = sorted({anchor + d for d in config.fit_offsets if 0 <= anchor + d < count})
    check = anchor + config.check_offset
    if check >= count:
        check = anchor - config.check_offset
    if not 0 <= check < count or check in fit:
        raise ValueError("no disjoint checking frame")
    return fit, check


def _transform(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def frame_cloud(frame, depth_scale, config):
    import cv2
    import open3d as o3d
    depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    rgb = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
    if depth is None or depth.ndim != 2 or depth.dtype != np.uint16 or rgb is None:
        raise ValueError("invalid RGB-D frame")
    # Input manifests follow the existing pipeline's pre-aligned RGB-D contract.
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != depth.shape:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    fx, fy, cx, cy = frame.intrinsics
    if frame.rotate_ccw:
        old_width = depth.shape[1]
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fx, fy, cx, cy = fy, fx, cy, old_width - 1.0 - cx
    step = config.pixel_stride
    vv, uu = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    z = depth[::step, ::step].astype(float) / depth_scale
    valid = (z >= 0.30) & (z <= 4.5) & np.isfinite(z)
    points = np.column_stack(((uu[valid]-cx)*z[valid]/fx, (vv[valid]-cy)*z[valid]/fy, z[valid]))
    colors = rgb[::step, ::step][valid].astype(float) / 255.0
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def anchor_clouds(manifest, poses, ordinals, config=LightweightConfig()):
    import open3d as o3d
    by_id = {f.frame_id: f for f in manifest.frames}
    fit_clouds, checks, evidence = [], [], []
    for anchor in ordinals:
        fit, check = frame_indices(anchor, len(poses), config)
        clouds = []
        binding = []
        for index in fit + [check]:
            frame = by_id[poses[index].frame_id]
            cloud = frame_cloud(frame, manifest.depth_scale, config)
            cloud.transform(np.linalg.inv(poses[anchor].t_world_camera) @ poses[index].t_world_camera)
            clouds.append(cloud)
            binding.append({"frame_id":frame.frame_id, "role":"check" if index==check else "fit",
                            "color_sha256":sha256_file(frame.color_path), "depth_sha256":sha256_file(frame.depth_path)})
        merged = o3d.geometry.PointCloud()
        for cloud in clouds[:-1]: merged += cloud
        merged = merged.voxel_down_sample(config.cloud_voxel_m)
        if len(merged.points) > config.maximum_points:
            merged = merged.select_by_index(np.linspace(0,len(merged.points)-1,config.maximum_points,dtype=int))
        check_cloud = clouds[-1].voxel_down_sample(config.cloud_voxel_m)
        fit_clouds.append(merged);checks.append(check_cloud)
        evidence.append({"anchor_ordinal":anchor,"frames":binding,"fit_points":len(merged.points),"check_points":len(check_cloud.points)})
    return fit_clouds, checks, evidence


def overlap_roi(source, target, initial, distance):
    """Two-sided support under the fixed initial pose, no GT or result-based crop."""
    from scipy.spatial import cKDTree
    sp, tp = np.asarray(source.points), np.asarray(target.points)
    if not len(sp) or not len(tp): return source.select_by_index([]), target.select_by_index([])
    moved = _transform(sp, initial)
    keep_s = cKDTree(tp).query(moved)[0] <= distance
    keep_t = cKDTree(moved).query(tp)[0] <= distance
    return source.select_by_index(np.flatnonzero(keep_s)), target.select_by_index(np.flatnonzero(keep_t))


def source_information(points_source, normals_target, rotation):
    """Right perturbation/source tangent [rotation, translation] Hessian."""
    normals_source = normals_target @ rotation
    jacobian = np.column_stack((np.cross(points_source, normals_source), normals_source))
    hessian = jacobian.T @ jacobian / max(len(jacobian), 1)
    return hessian + np.eye(6) * max(np.trace(hessian), 1e-8) * 1e-8


def _heldout_direction(source, target, initial, refined, config):
    from scipy.spatial import cKDTree
    import open3d as o3d
    if min(len(source.points),len(target.points)) < config.minimum_points:
        return {"passes":False,"reason":"insufficient_check_cloud"}
    target = copy.deepcopy(target)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.10,max_nn=30))
    sp, tp, normals = np.asarray(source.points), np.asarray(target.points), np.asarray(target.normals)
    tree = cKDTree(tp)
    before_dist, before_idx = tree.query(_transform(sp,initial))
    fixed = before_dist <= 0.10
    if int(fixed.sum()) < config.minimum_points:
        return {"passes":False,"reason":"insufficient_fixed_check_support","fixed_support":int(fixed.sum())}
    sp = sp[fixed]
    before_dist, before_idx = before_dist[fixed], before_idx[fixed]
    moved = _transform(sp,refined)
    after_dist, after_idx = tree.query(moved)
    before = np.abs(np.sum((_transform(sp,initial)-tp[before_idx])*normals[before_idx],axis=1))
    after = np.abs(np.sum((moved-tp[after_idx])*normals[after_idx],axis=1))
    old = float(np.sqrt(np.mean(np.minimum(before,0.15)**2)))
    new = float(np.sqrt(np.mean(np.minimum(after,0.15)**2)))
    coverage = float(np.mean(after_dist<=0.10))
    # Geometry agreement on a frame excluded from that anchor's fitting cloud.
    passes = old>1e-5 and new <= old*(1-config.minimum_validation_improvement) and coverage>=config.minimum_coverage_retention
    return {"passes":bool(passes),"fixed_support":len(sp),"before_plane_rmse_m":old,
            "after_plane_rmse_m":new,"coverage_retention":coverage,
            "before_nn_rmse_m":float(np.sqrt(np.mean(before_dist**2))),
            "after_nn_rmse_m":float(np.sqrt(np.mean(np.minimum(after_dist,0.15)**2))) }


def refine_pair(source, target, check_source, check_target, initial, *, roi=False,
                colored=False, config=LightweightConfig()):
    import open3d as o3d
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    started=time.perf_counter()
    initial=validate_se3(initial)
    fit_s, fit_t = overlap_roi(source,target,initial,config.roi_distance_m) if roi else (source,target)
    result={"roi":roi,"colored_multiscale":colored,"fit_source_points":len(fit_s.points),
            "fit_target_points":len(fit_t.points),"gt_consumed":False,"accepted":False}
    if min(len(fit_s.points),len(fit_t.points))<config.minimum_points:
        return {**result,"reason":"insufficient_roi_support","runtime_s":time.perf_counter()-started}
    transform=initial.copy()
    scales=((0.10,20),(0.05,15),(0.025,10)) if colored else ((0.05,30),)
    try:
        for voxel,iterations in scales:
            a,b=fit_s.voxel_down_sample(voxel),fit_t.voxel_down_sample(voxel)
            for cloud in (a,b):cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2,max_nn=30))
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
            if colored:
                reg=o3d.pipelines.registration.registration_colored_icp(a,b,voxel*1.5,transform,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),criteria)
            else:
                reg=o3d.pipelines.registration.registration_icp(a,b,voxel*1.5,transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),criteria)
            transform=validate_se3(reg.transformation)
        delta=transform @ np.linalg.inv(initial)
        dt=float(np.linalg.norm(delta[:3,3]));dr=float(np.linalg.norm(Rotation.from_matrix(delta[:3,:3]).as_rotvec())*180/np.pi)
        result.update(update_translation_m=dt,update_rotation_deg=dr,transform=transform.tolist())
        if dt>config.maximum_update_translation_m or dr>config.maximum_update_rotation_deg:
            return {**result,"reason":"update_limit","runtime_s":time.perf_counter()-started}
        forward=_heldout_direction(check_source,check_target,initial,transform,config)
        reverse=_heldout_direction(check_target,check_source,np.linalg.inv(initial),np.linalg.inv(transform),config)
        result.update(check_forward=forward,check_reverse=reverse)
        if not forward['passes'] or not reverse['passes']:
            return {**result,"reason":"heldout_check_failed","runtime_s":time.perf_counter()-started}
        # This is geometry-derived directional information; never a calibrated probability.
        b=copy.deepcopy(fit_t);b.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.10,max_nn=30))
        sp=np.asarray(fit_s.points);tp=np.asarray(b.points)
        distances,indices=cKDTree(tp).query(_transform(sp,transform));mask=distances<=0.05
        if int(mask.sum())<config.minimum_points:
            return {**result,"reason":"insufficient_information_support","runtime_s":time.perf_counter()-started}
        info=source_information(sp[mask],np.asarray(b.normals)[indices[mask]],transform[:3,:3])
        result.update(accepted=True,reason="bounded_independent_depth_check",information=info.tolist())
    except (RuntimeError,ValueError) as error:
        result.update(reason="registration_failed",error=str(error))
    return {**result,"runtime_s":time.perf_counter()-started}
