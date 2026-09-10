import os
import random
import time
from dataclasses import dataclass

import hydra
import numpy as np
import open3d as o3d
import torch
from omegaconf import OmegaConf
from lygra.contact_set import get_link_dependency_matrix
from lygra.kinematics import build_kinematics_tree
from lygra.memory import IKGPUBufferPool
from lygra.mesh import (
    get_urdf_mesh,
    get_urdf_mesh_decomposed,
    get_urdf_mesh_for_projection,
    trimesh_to_open3d,
)
from lygra.mesh_analyzer import get_support_point_mask
from lygra.object_rules import (
    apply_object_contact_rules,
    compile_object_contact_rules,
    exclusive_contact_mask,
)
from lygra.robot import build_robot
from lygra.utils.geom_utils import MeshObject
from lygra.utils.robot_visualizer import RobotVisualizer

from lygra.pipeline.module.contact_collection import (
    sample_pose_and_contact_from_interaction,
)
from lygra.pipeline.module.contact_optimization import search_contact_point
from lygra.pipeline.module.contact_query import (
    batch_object_all_contact_fields_interaction,
)
from lygra.pipeline.module.kinematics import batch_contact_adjustment, batch_ik
from lygra.pipeline.module.object_placement import (
    get_object_pose_sampling_args,
    sample_object_pose,
)
from lygra.pipeline.module.postprocess import batch_assign_free_finger_and_filter

@dataclass
class ContactRuntime:
    field: object
    parent_ids: torch.Tensor
    dependency_matrix: torch.Tensor
    accel_structure: object


@dataclass
class ObjectRuntime:
    points: torch.Tensor
    normals: torch.Tensor
    points_all: torch.Tensor
    point_labels: torch.Tensor
    rules: object


def _build_contact_runtime(robot, digits_cfg, tree, dependency_sets, cf_accel):
    contact_field = robot.get_contact_field(digits_cfg)
    contact_parent_links = contact_field.get_all_parent_link_names()
    contact_parent_ids = [tree.get_link_id(link) for link in contact_parent_links]
    contact_parent_ids = torch.tensor(contact_parent_ids).cuda()
    dependency_matrix = get_link_dependency_matrix(contact_field, dependency_sets).cuda()
    accel_structure = contact_field.generate_acceleration_structure(method=cf_accel)
    return ContactRuntime(
        field=contact_field,
        parent_ids=contact_parent_ids,
        dependency_matrix=dependency_matrix,
        accel_structure=accel_structure,
    )


def _sample_object_runtime(
    mesh_object,
    n_sample_point,
    object_point_sampling_strategy,
    object_point_filter,
    parts_cfg,
    config,
    contact_link_names,
):
    (points, normals, point_labels), name2id = mesh_object.sample_point_and_normal(
        count=n_sample_point,
        sample_policy=object_point_sampling_strategy,
        sample_weights=parts_cfg.get('sample_weights'),
        centroid_sample_scale=parts_cfg.get('centroid_sample_scale', {})
    )
    points_all = torch.from_numpy(points).cuda().float()
    normals_all = torch.from_numpy(normals).cuda().float()
    point_labels = torch.from_numpy(point_labels).cuda().long()
    use_part_rules = bool(parts_cfg)

    if object_point_filter and use_part_rules:
        support_point_mask = get_support_point_mask(points_all, normals_all, [0.01])[0]
        points = points_all[torch.where(support_point_mask)]
        normals = normals_all[torch.where(support_point_mask)]
        point_labels = point_labels[torch.where(support_point_mask)]
    else:
        points = points_all
        normals = normals_all

    specific_link_names = set(config["specific_link"])
    rules = compile_object_contact_rules(
        parts_cfg=parts_cfg,
        point_labels=point_labels,
        name_to_id=name2id,
        contact_link_names=contact_link_names,
        specific_link_names=specific_link_names,
    )

    return ObjectRuntime(
        points=points,
        normals=normals,
        points_all=points_all,
        point_labels=point_labels,
        rules=rules,
    )


def _get_object_parts_cfg(parts_config, robot_name):
    parts_cfg = OmegaConf.to_container(parts_config, resolve=True)
    robot_rules = parts_cfg.pop("robot_rules", None)
    if robot_rules is None:
        return parts_cfg
    if robot_name not in robot_rules:
        raise ValueError(
            f"Object part rules have no configuration for robot '{robot_name}'. "
            f"Add parts.robot_rules.{robot_name} to the object YAML file."
        )
    parts_cfg.update(robot_rules[robot_name])
    return parts_cfg


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    visualize = cfg.visualize
    robot_name = cfg.robot
    category = cfg.object.asset.category
    object_id = cfg.object.asset.instance_id
    asset_root = cfg.object.asset.root
    input_dir = cfg.object.asset.get("input_dir", os.path.join(asset_root, category))
    output_category = os.path.basename(os.path.normpath(input_dir)) or category
    scale_factor = cfg.object.asset.scale_factor
    object_mesh_path = os.path.join(input_dir, object_id, "labeled.obj")
    batch_size_outer = cfg.batch_size_outer
    batch_size_inner = cfg.batch_size_inner
    accept_less_contact = cfg.accept_less_contact
    n_sample_point = cfg.n_sample_point
    ik_finetune_iter = cfg.ik_finetune_iter
    zo_lr_sigma = cfg.zo_lr_sigma
    contact_score_threshold = cfg.contact_score_threshold
    cf_accel = cfg.cf_accel
    object_pose_sampling_strategy = cfg.object_pose_sampling_strategy
    object_point_sampling_strategy = cfg.object_point_sampling_strategy
    object_point_filter = cfg.object_point_filter
    sample_pose_failure_resample_threshold = cfg.get("sample_pose_failure_resample_threshold", 10)
    target_n_result = cfg.target_n_result
    save_root = cfg.save_dir
    #------------------------------------------
    if robot_name not in cfg.object.digits:
        raise ValueError(
            f"Object '{category}' has no digit configuration for robot "
            f"'{robot_name}'. Add object.digits.{robot_name} to its YAML file."
        )
    digits_cfg = OmegaConf.to_container(
        cfg.object.digits[robot_name],
        resolve=True,
    )
    n_contact = digits_cfg["n_contact"]
    parts_cfg = _get_object_parts_cfg(cfg.object.parts, robot_name)

    robot = build_robot(robot_name)

    # Robot Structure.
    tree = build_kinematics_tree(
        urdf_path=robot.urdf_path,
        active_joint_names=robot.get_active_joints(),
    )
    tree.set_fixed_zero_joints(
        digits_cfg.get("fixed_zero_joints", digits_cfg.get("disabled_joints", []))
    )

    # Robot Mesh Data
    mesh_data = get_urdf_mesh(
        urdf_path=robot.urdf_path,
        tree=tree,
        mesh_scale=robot.get_mesh_scale()
    )

    config = robot.get_contact_field_config(digits_cfg)
    mesh_data_for_ik = get_urdf_mesh_for_projection(
        urdf_path=robot.urdf_path,
        tree=tree,
        config=config,
        mesh_scale=robot.get_mesh_scale()
    )

    decomposed_static_mesh_data = get_urdf_mesh_decomposed(
        urdf_path=robot.urdf_path,
        tree=tree,
        override_link_names=robot.get_static_links(),
        mesh_scale=robot.get_mesh_scale()
    )

    decomposed_mesh_data = get_urdf_mesh_decomposed(
        urdf_path=robot.urdf_path,
        tree=tree,
        mesh_scale=robot.get_mesh_scale()
    )

    # Robot Collision & Kinematics Metadata
    self_collision_link_pairs = tree.get_self_collision_check_link_pairs(
        link_body_id=decomposed_mesh_data['link_body_id'],
        whitelist_link=[]
    )

    self_collision_link_pairs = torch.from_numpy(self_collision_link_pairs).cuda().int()

    dependency_sets = tree.get_dependency_sets([robot.get_base_link()])
    contact_runtime = _build_contact_runtime(
        robot,
        digits_cfg,
        tree,
        dependency_sets,
        cf_accel,
    )

    # Object Data.
    mesh_object = MeshObject(object_mesh_path, scale_factor=scale_factor)
    object_area = mesh_object.get_area()
    zo_lr = ((object_area / n_sample_point) ** 0.5) * zo_lr_sigma
    object_runtime = _sample_object_runtime(
        mesh_object,
        n_sample_point,
        object_point_sampling_strategy,
        object_point_filter,
        parts_cfg,
        config,
        contact_runtime.field.all_patch_parent_link_names,
    )

    # IK GPU buffer. 
    gpu_memory_pool = IKGPUBufferPool(
        n_dof=tree.n_dof(), 
        n_link=tree.n_link(), 
        max_batch=min(batch_size_outer * batch_size_inner, 65536),
        retry=10
    )

    # ---------------
    # Inference Stage 
    # ---------------

    print("Launch Inference")
    candidate_qpos = []
    candidate_poses = []
    poses_num = 0
    sample_pose_failure_count = 0
    t1 = time.time()
    while True:
        with torch.no_grad():
            object_poses, condition = sample_object_pose(
                n=batch_size_outer, 
                points=object_runtime.points,
                normals=object_runtime.normals,
                contact_field=contact_runtime.field,
                tree=tree, 
                mesh_data=decomposed_static_mesh_data,
                sampling_args=get_object_pose_sampling_args(
                    object_pose_sampling_strategy,
                    object_runtime.rules.specific_touchable_mask,
                    use_static_prob=0.0,
                    robot=robot,
                ),
            )

            # Contact Field BVH Traversal
            interaction_matrix_hand_point_idx = batch_object_all_contact_fields_interaction(
                object_pos=object_runtime.points,
                object_normal=object_runtime.normals,
                object_pose=object_poses, 
                accel_structure=contact_runtime.accel_structure,
            )  # num_poses , num_patches , num_points

            apply_object_contact_rules(
                interaction_matrix_hand_point_idx,
                contact_runtime.field.all_patch_parent_link_names,
                object_runtime.rules,
            )
            
            interaction_matrix = (interaction_matrix_hand_point_idx >= 0).int()
            link_interaction_matrix, thumb_indices = contact_runtime.field.reduce_link_interaction(
                interaction_matrix
            )
            # Get Contact Domain
            try:
                contact_domain_pos, contact_domain_normal, contact_domain_point_idx, \
                object_poses, contact_link_ids, condition, valid_outer_idx = \
                sample_pose_and_contact_from_interaction(
                    n_contact=n_contact,
                    interaction_matrix=link_interaction_matrix, 
                    dependency_matrix=contact_runtime.dependency_matrix,
                    object_points=object_runtime.points,
                    object_normals=object_runtime.normals,
                    object_poses=object_poses,
                    condition=condition,
                    thumb_indices=thumb_indices,
                    accept_downsamle=accept_less_contact,
                )
            except SystemExit:
                print("sample_pose_and_contact_from_interaction failed")
                sample_pose_failure_count += 1
                if (
                    sample_pose_failure_resample_threshold is not None
                    and sample_pose_failure_resample_threshold > 0
                    and sample_pose_failure_count >= sample_pose_failure_resample_threshold
                ):
                    print(
                        "Resampling contact field and object points after "
                        f"{sample_pose_failure_count} consecutive sample_pose_and_contact_from_interaction failures"
                    )
                    del contact_runtime
                    torch.cuda.empty_cache()

                    contact_runtime = _build_contact_runtime(
                        robot,
                        digits_cfg,
                        tree,
                        dependency_sets,
                        cf_accel,
                    )
                    object_runtime = _sample_object_runtime(
                        mesh_object,
                        n_sample_point,
                        object_point_sampling_strategy,
                        object_point_filter,
                        parts_cfg,
                        config,
                        contact_runtime.field.all_patch_parent_link_names,
                    )
                    torch.cuda.empty_cache()
                    sample_pose_failure_count = 0
                continue
            sample_pose_failure_count = 0
            
            # Search Contact Points in Contact Domain
            target_contact_pos, target_contact_normal, target_contact_point_idx, \
            object_poses, target_contact_link_ids, target_batch_outer_ids = \
            search_contact_point(
                contact_domain_pos=contact_domain_pos, 
                contact_domain_normal=contact_domain_normal, 
                contact_domain_point_idx=contact_domain_point_idx,
                object_poses=object_poses, 
                contact_ids=contact_link_ids,
                batch_size=batch_size_inner,
                return_hand_frame=True,
                condition=condition,
                zo_lr=zo_lr,
                threshold=contact_score_threshold
            )
            if target_contact_pos.shape[0] == 0 :
                print(f"search_contact_point failed")
                continue

            exclusive_mask = exclusive_contact_mask(
                target_contact_link_ids,
                target_contact_point_idx,
                object_runtime.point_labels,
                object_runtime.rules.exclusive_part_ids,
            )
            if not exclusive_mask.all():
                keep_idx = torch.where(exclusive_mask)
                target_contact_pos = target_contact_pos[keep_idx]
                target_contact_normal = target_contact_normal[keep_idx]
                target_contact_point_idx = target_contact_point_idx[keep_idx]
                object_poses = object_poses[keep_idx]
                target_contact_link_ids = target_contact_link_ids[keep_idx]
                target_batch_outer_ids = target_batch_outer_ids[keep_idx]
                if target_contact_pos.shape[0] == 0:
                    print("exclusive_touchable_area filtered all solutions")
                    continue

            contact_ids, local_contact_ids = contact_runtime.field.sample_contact_ids(
                interaction_matrix=interaction_matrix[valid_outer_idx], 
                interaction_matrix_hand_point_idx=interaction_matrix_hand_point_idx[valid_outer_idx],
                target_batch_outer_ids=target_batch_outer_ids, 
                target_contact_link_ids=target_contact_link_ids, 
                target_contact_point_idx=target_contact_point_idx
            )
            contact_pos_in_linkf, contact_normal_in_linkf = (
                contact_runtime.field.sample_contact_geometry(
                    contact_ids,
                    local_contact_ids,
                )
            )

            # Kinematics Optimization (I)
            # Coarse IK. Might not align well.
            result = batch_ik(
                tree=tree,
                contact_ids=contact_ids,
                contact_parent_ids=contact_runtime.parent_ids,
                contact_pos_in_linkf=contact_pos_in_linkf.float(),
                contact_normal_in_linkf=contact_normal_in_linkf.float(),
                target_contact_pos=target_contact_pos.float(),
                target_contact_normal=target_contact_normal.float(),
                object_pose=object_poses.float(),
                gpu_memory_pool=gpu_memory_pool
            )
            n1 = len(result['q'])
            if n1 <= 0:
                print(f'batch_ik has no solutions')
                continue
            # Kinematics Optimization (II)
            # Finegrained Finger Pose by Iterative Projection + IK Adjustment.
            result = batch_contact_adjustment(
                tree=tree,
                mesh=mesh_data_for_ik,
                q_init=result["q"],
                q_mask=result["q_mask"],
                contact_ids=contact_ids,
                contact_link_ids=result["contact_link_id"],
                contact_pos_in_linkf=result["contact_pos"],
                contact_normal_in_linkf=result["contact_normal"],
                target_contact_pos=result["target_pos"],
                target_contact_normal=result["target_normal"],
                object_pose=result["object_pose"],
                n_iter=ik_finetune_iter,
                gpu_memory_pool=gpu_memory_pool,
                ret_mesh_buffer=True
            )
            n2 = len(result['q'])
            if n2 <= 0:
                print(f'batch_contact_adjustment filtered out {n1-n2} solutions')
                continue
            # Postprocessing: 
            # Search Free Finger Configuration & Remove Invalid Results (collision).
            # Hand-to-hand  -- AABB broad phase + GJK narrow phase 
            # Hand-to-point -- AABB broad phase + halfplane-test narrow phase
            result = batch_assign_free_finger_and_filter(
                tree=tree,
                result=result,
                object_point=object_runtime.points_all,
                self_collision_link_pairs=self_collision_link_pairs,
                decomposed_mesh_data=decomposed_mesh_data
            )
            n3 = len(result['q'])
            if n3 <= 0:
                print(f'batch_assign_free_finger_and_filter filtered out {n2-n3} solutions')
                continue

        poses_num += len(result['object_pose'])
        candidate_qpos.append(result['q'].detach().cpu().numpy())
        candidate_poses.append(result['object_pose'].cpu().numpy())
        print("Number of Solutions:", poses_num)
        if poses_num >= target_n_result:
            candidate_qpos = np.concatenate(candidate_qpos, axis=0)
            candidate_poses = np.concatenate(candidate_poses, axis=0)
            save_dir = os.path.join(save_root, robot_name, output_category, object_id)
            os.makedirs(save_dir, exist_ok=True)
            np.save(
                os.path.join(save_dir, "qpos.npy"),
                candidate_qpos[:target_n_result, ...],
            )
            np.save(
                os.path.join(save_dir, "opos.npy"),
                candidate_poses[:target_n_result, ...],
            )
            print(f"{target_n_result} poses generated...{save_dir}")
            break

    # -----------------
    # Visualize Results
    # -----------------
    if not visualize:
        return

    viewer = RobotVisualizer(robot)
    t2 = time.time()
    print(f"Inference Time: {t2-t1:.2f} seconds for {poses_num} poses.")
    while True:
        idx = random.randint(0, poses_num - 1)
        robot_mesh = viewer.get_mesh_fk(candidate_qpos[idx:idx+1], visual=True)

        object_mesh = mesh_object.mesh.copy()
        object_mesh.apply_transform(candidate_poses[idx])
        object_mesh_o3d = trimesh_to_open3d(object_mesh)
        
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        material.base_color = [245 / 256, 162 / 256, 98 / 256, 0.8]
        material.base_metallic = 0.0
        material.base_roughness = 1.0
        object_mesh = {"name": 'object', "geometry": object_mesh_o3d, "material": material}
        viewer.show(robot_mesh + [object_mesh])

        if input("Continue? (Y/n)") in ['n', 'N']:
            break


if __name__ == "__main__":
    main()
