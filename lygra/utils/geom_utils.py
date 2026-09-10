# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import trimesh
import numpy as np


# class MeshObject:
#     def __init__(self, obj_path):
#         self.mesh = trimesh.load(obj_path)

#     def sample_point_and_normal(self, count=2000):
#         points, face_indices = trimesh.sample.sample_surface(self.mesh, count=count)
#         normals = self.mesh.face_normals[face_indices]
#         return points, normals

#     def get_area(self):
#         return self.mesh.area


class MeshObject:
    def __init__(self, obj_path, scale_factor=1.0):
        self.scale_factor = scale_factor

        loaded = trimesh.load(obj_path)

        if isinstance(loaded, trimesh.Scene):
            self.geometries = list(loaded.geometry.values())
        else:
            self.geometries = [loaded]

        # 对每个子 mesh 做 scale（这是关键）
        if scale_factor != 1.0:
            for g in self.geometries:
                g.apply_scale(scale_factor)

        # 合并 mesh（此时已经是缩放后的）
        self.mesh = trimesh.util.concatenate(self.geometries)

        print(f"检测到 {len(self.geometries)} 个独立零件，scale={scale_factor}")

    def sample_point_and_normal(
        self,
        count=2048,
        sample_policy="area_weighted",
        sample_weights=None,
        centroid_sample_scale=None,
    ):
        """
        根据零件分别采样，并返回 point_labels
        Label 0: Part 1 (通常是四指区域)
        Label 1: Part 2 (通常是拇指区域)
        """
        if not sample_weights:
            points, face_indices = trimesh.sample.sample_surface(self.mesh, count=count)
            normals = self.mesh.face_normals[face_indices]
            labels = np.zeros(len(points), dtype=np.int32)
            return (points, normals, labels), {}

        all_points = []
        all_normals = []
        all_labels = []
        name2id = {}
        centroid_sample_scale = centroid_sample_scale or {}
        # 简单分配采样配额（也可以根据面积比例分配）
        # 假设我们有两个零件：[Part1, Part2]
        n_parts = len(self.geometries)
        count_cum = 0
        # count_per_part = count // n_parts
        for i, geo in enumerate(self.geometries):
            mat = geo.visual.material
            name = getattr(mat, "name", None)
            if name not in sample_weights.keys():
                continue
            # if i == len(self.geometries) - 1:
            #     count_per_part = count - count_cum
            #     break
            if i == len(self.geometries):
                count_per_part = count - count_cum
                break
            if sample_policy == "manual_weighted":
                count_per_part = int(count * sample_weights[name])
            elif sample_policy == "area_weighted":
                count_per_part = int(count * geo.area / self.mesh.area)
            else:
                raise ValueError(
                    f"Unknown sample_policy: {sample_policy}!available choice: 'area_weighted' or 'manual_weighted'"
                )
            count_cum += count_per_part
            pts, face_idx = trimesh.sample.sample_surface(geo, count=count_per_part)
            normals = geo.face_normals[face_idx]
            if name in centroid_sample_scale:
                s = float(centroid_sample_scale[name])
                center = geo.centroid
                pts = center + (pts - center) * s
            all_points.append(pts)
            all_normals.append(normals)
            name2id[name] = i
            # 给这批点打上标签 i
            all_labels.append(np.full(len(pts), i, dtype=np.int32))
        return (
            np.concatenate(all_points),
            np.concatenate(all_normals),
            np.concatenate(all_labels),
        ), name2id

    def get_area(self):
        return self.mesh.area


# def get_tangent_plane(batch_vector):
#     ''' Get the tangent planes of vectors.

#     Args:
#         batch_vector: [..., 3] (torch.Tensor or np.ndarray)

#     Returns:
#         x:            [..., 3]
#         y:            [..., 3]
#     '''
#     if isinstance(batch_vector, torch.Tensor):
#         shape = batch_vector.shape[:-1]
#         batch_vector = F.normalize(batch_vector.reshape(-1, 3), dim=-1)
#         x = batch_vector + torch.ones_like(batch_vector) * 2
#         y = torch.cross(batch_vector, F.normalize(x, dim=-1), dim=-1)
#         x = torch.cross(y, batch_vector, dim=-1)

#         return x.reshape(*shape, 3), y.reshape(*shape, 3)
#     else:
#         # numpy
#         shape = batch_vector.shape[:-1]

#         batch_vector = np.reshape(batch_vector, (-1, 3))
#         norm = np.linalg.norm(batch_vector, axis=-1, keepdims=True)
#         batch_vector = batch_vector / np.clip(norm, 1e-8, None)

#         x = batch_vector + 2.0  # broadcast with scalar
#         x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
#         x = x / np.clip(x_norm, 1e-8, None)

#         y = np.cross(batch_vector, x)
#         y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
#         y = y / np.clip(y_norm, 1e-8, None)

#         x = np.cross(y, batch_vector)
#         x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
#         x = x / np.clip(x_norm, 1e-8, None)


#         return x.reshape(*shape, 3), y.reshape(*shape, 3)


def get_tangent_plane(batch_vector):
    """获取向量的切平面（具有强随机性）。

    Args:
        batch_vector: [..., 3] (torch.Tensor 或 np.ndarray)

    Returns:
        x:            [..., 3]
        y:            [..., 3]
    """
    if isinstance(batch_vector, torch.Tensor):
        shape = batch_vector.shape[:-1]
        # 1. 归一化 Z 轴
        z = F.normalize(batch_vector.reshape(-1, 3), dim=-1)

        # 2. 生成随机种子向量并归一化
        rand_vec = torch.randn_like(z)
        rand_vec = F.normalize(rand_vec, dim=-1)

        # 3. 通过两次叉乘构建正交基
        y = torch.cross(z, rand_vec, dim=-1)
        # 检查极小概率的共线情况
        norms = torch.norm(y, dim=-1, keepdim=True)
        mask = (norms < 1e-4).squeeze()
        if mask.any():
            fallback = torch.tensor([1.0, 0.0, 0.0], device=z.device)
            y[mask] = torch.cross(z[mask], fallback.expand_as(z[mask]), dim=-1)

        y = F.normalize(y, dim=-1)
        x = torch.cross(y, z, dim=-1)

        return x.reshape(*shape, 3), y.reshape(*shape, 3)

    else:
        # numpy 分支
        shape = batch_vector.shape[:-1]
        z = np.reshape(batch_vector, (-1, 3))

        # 1. 归一化 Z 轴
        z_norm = np.linalg.norm(z, axis=-1, keepdims=True)
        z = z / np.clip(z_norm, 1e-8, None)

        # 2. 生成随机种子向量 (使用 standard_normal 保证方向分布均匀)
        rand_vec = np.random.standard_normal(z.shape)
        rand_vec /= np.clip(
            np.linalg.norm(rand_vec, axis=-1, keepdims=True), 1e-8, None
        )

        # 3. 计算 Y = Z x Rand
        y = np.cross(z, rand_vec)

        # 4. 检查共线并归一化 Y
        y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
        mask = (y_norm < 1e-4).flatten()
        if np.any(mask):
            fallback = np.array([1.0, 0.0, 0.0])
            y[mask] = np.cross(z[mask], fallback)
            y_norm[mask] = np.linalg.norm(y[mask], axis=-1, keepdims=True)

        y = y / np.clip(y_norm, 1e-8, None)

        # 5. 计算 X = Y x Z
        x = np.cross(y, z)
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
        x = x / np.clip(x_norm, 1e-8, None)

        return x.reshape(*shape, 3), y.reshape(*shape, 3)
