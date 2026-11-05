"""孔位编号追踪服务。

针对「一圈孔 + 无基准特征」的场景，采用：

1. 首帧按极角排序建立标准编号（锚定）；
2. 后续帧通过圆形角度最近邻（匈牙利算法）关联，维持 ``hole_id`` 稳定；
3. 短暂漏检的孔保留编号并标记 ``missing``，超过阈值后丢弃。

注意：追踪状态保存在进程内存中，默认单 worker 部署；若启用多 worker，
需替换为 Redis 等共享存储以跨进程保持一致。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


class HoleTracker:
    """单个工件的一圈孔位编号追踪器。"""

    def __init__(
        self,
        max_angle_diff: float = 45.0,
        max_missed_frames: int = 3,
    ) -> None:
        self._max_angle_diff = max_angle_diff
        self._max_missed_frames = max_missed_frames
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1
        # 最近一次成功拟合圆得到的圆心，孔数不足时复用它计算极角
        self._center_x: Optional[float] = None
        self._center_y: Optional[float] = None

    def reset(self) -> None:
        """清空状态，换新工件时调用。"""
        self._tracks.clear()
        self._next_id = 1
        self._center_x = None
        self._center_y = None

    @property
    def is_initialized(self) -> bool:
        return bool(self._tracks)

    def update(
        self,
        holes: List[Dict[str, float]],
        center_x: float,
        center_y: float,
    ) -> List[Dict[str, Any]]:
        """输入当前帧螺孔与圆心，输出带稳定 ``hole_id`` 的孔列表。"""
        if not holes:
            return self._on_miss_all()

        self._center_x, self._center_y = center_x, center_y
        observations = self._build_observations(holes, center_x, center_y)

        if not self._tracks:
            return self._initialize(observations)

        return self._associate(observations)

    def update_nearest(
        self,
        holes: List[Dict[str, float]],
        image_w: float,
        image_h: float,
    ) -> List[Dict[str, Any]]:
        """孔数不足（< 3）时的降级追踪：只跟踪离图像中心最近的孔并返回其编号。

        无法拟合圆时，沿用最近一次成功拟合的圆心（无则用图像中心）计算极角，
        与已有轨迹按极角匹配取编号。该路径是只读的，不改动已有轨迹状态，
        以免干扰常规（可拟合圆）帧的关联。
        """
        if not holes:
            return []

        img_cx, img_cy = image_w / 2.0, image_h / 2.0
        nearest = min(
            holes,
            key=lambda h: (h["center_x"] - img_cx) ** 2 + (h["center_y"] - img_cy) ** 2,
        )

        ref_x = self._center_x if self._center_x is not None else img_cx
        ref_y = self._center_y if self._center_y is not None else img_cy
        observation = self._build_observations([nearest], ref_x, ref_y)[0]

        if not self._tracks:
            # 尚无任何轨迹：锚定该孔，保证可以返回编号
            target_id = self._next_id
            self._next_id += 1
            self._tracks[target_id] = self._track_from_obs(observation)
            return [self._format(target_id, self._tracks[target_id])]

        target_id = self._match_by_angle(observation["angle"])
        if target_id is None:
            # 无落在容忍角度内的轨迹：取极角最接近者作为最佳估计（不新增编号）
            target_id = min(
                self._tracks,
                key=lambda tid: self._angle_diff(self._tracks[tid]["angle"], observation["angle"]),
            )
        return [self._format(target_id, self._track_from_obs(observation))]

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """两个极角之间的最小圆周角差（0~180）。"""
        diff = abs(a - b) % 360.0
        return min(diff, 360.0 - diff)

    def _match_by_angle(self, angle: float) -> Optional[int]:
        """在已有轨迹中找极角最接近的孔，超出容忍角度则视为未匹配。"""
        best_id: Optional[int] = None
        best_diff: Optional[float] = None
        for tid, track in self._tracks.items():
            diff = self._angle_diff(track["angle"], angle)
            if best_diff is None or diff < best_diff:
                best_id, best_diff = tid, diff
        if best_diff is not None and best_diff <= self._max_angle_diff:
            return best_id
        return None

    # ------------------------------------------------------------------
    def _build_observations(
        self,
        holes: List[Dict[str, float]],
        center_x: float,
        center_y: float,
    ) -> List[Dict[str, Any]]:
        observations = []
        for h in holes:
            angle = math.degrees(
                math.atan2(h["center_y"] - center_y, h["center_x"] - center_x)
            ) % 360.0
            observations.append(
                {
                    "conf": h["conf"],
                    "center_x": h["center_x"],
                    "center_y": h["center_y"],
                    "angle": angle,
                }
            )
        return observations

    def _initialize(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 首帧锚定：按极角升序编号
        observations = sorted(observations, key=lambda o: o["angle"])
        self._tracks = {}
        for obs in observations:
            self._tracks[self._next_id] = self._track_from_obs(obs)
            self._next_id += 1
        return self._snapshot()

    def _associate(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        track_ids = sorted(self._tracks)
        track_angles = [self._tracks[i]["angle"] for i in track_ids]
        obs_angles = [o["angle"] for o in observations]

        n_tracks = len(track_ids)
        n_obs = len(observations)
        cost = np.zeros((n_tracks, n_obs))
        for a, ta in enumerate(track_angles):
            for b, oa in enumerate(obs_angles):
                diff = abs(ta - oa)
                cost[a, b] = min(diff, 360.0 - diff)  # 圆形角度差

        row, col = linear_sum_assignment(cost)

        matched_track_idx: set = set()
        matched_obs_idx: set = set()
        new_tracks: Dict[int, Dict[str, Any]] = {}

        for a, b in zip(row, col):
            if cost[a, b] <= self._max_angle_diff:
                new_tracks[track_ids[a]] = self._track_from_obs(observations[b])
                matched_track_idx.add(a)
                matched_obs_idx.add(b)

        # 未匹配的观测 -> 新孔
        for b, obs in enumerate(observations):
            if b not in matched_obs_idx:
                new_tracks[self._next_id] = self._track_from_obs(obs)
                self._next_id += 1

        # 未匹配的旧 track -> 记一次漏检
        for a, tid in enumerate(track_ids):
            if a not in matched_track_idx:
                old = self._tracks[tid]
                misses = old.get("misses", 0) + 1
                if misses <= self._max_missed_frames:
                    new_tracks[tid] = {**old, "misses": misses}

        self._tracks = new_tracks
        return self._snapshot()

    def _on_miss_all(self) -> List[Dict[str, Any]]:
        new_tracks = {}
        for tid, track in self._tracks.items():
            misses = track.get("misses", 0) + 1
            if misses <= self._max_missed_frames:
                new_tracks[tid] = {**track, "misses": misses}
        self._tracks = new_tracks
        return self._snapshot()

    @staticmethod
    def _track_from_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "conf": obs["conf"],
            "center_x": obs["center_x"],
            "center_y": obs["center_y"],
            "angle": obs["angle"],
            "misses": 0,
        }

    @staticmethod
    def _format(hole_id: int, track: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "hole_id": hole_id,
            "conf": round(track["conf"], 3),
            "center_x": float(track["center_x"]),
            "center_y": float(track["center_y"]),
            "angle_deg": round(track["angle"], 2),
            "missing": track.get("misses", 0) > 0,
        }

    def _snapshot(self) -> List[Dict[str, Any]]:
        return [self._format(tid, self._tracks[tid]) for tid in sorted(self._tracks)]


class TrackerRegistry:
    """按 ``namespace`` + ``session_id`` 隔离多个工件的追踪器。

    ``namespace`` 用于区分不同检测模型（如 ``usb_hole7`` / ``usb_hole11``），
    避免同一 ``session_id`` 下的追踪状态互相串用。
    """

    def __init__(self) -> None:
        self._trackers: Dict[str, HoleTracker] = {}

    @staticmethod
    def _key(namespace: str, session_id: str) -> str:
        return f"{namespace}:{session_id}"

    def get(self, namespace: str, session_id: str) -> HoleTracker:
        key = self._key(namespace, session_id)
        if key not in self._trackers:
            self._trackers[key] = HoleTracker()
        return self._trackers[key]

    def reset(self, namespace: str, session_id: str) -> None:
        self._trackers.pop(self._key(namespace, session_id), None)


# 进程级单例（单 worker 部署）
tracker_registry = TrackerRegistry()
