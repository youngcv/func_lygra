# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.kinematics import build_kinematics_tree
from lygra.contact_field import build_contact_field
from abc import ABC, abstractmethod
import numpy as np


class RobotInterface(ABC):
    def __init__(self, urdf_path=None):
        self.urdf_path = urdf_path
        if self.urdf_path is None:
            self.urdf_path = self.get_default_urdf_path()
            assert self.urdf_path is not None, "Must specify URDF."

        self.tree = None
        return 

    def get_default_urdf_path(self):
        return None

    def get_canonical_space(self):
        return None

    @abstractmethod
    def get_active_joints(self):
        pass

    @abstractmethod
    def get_base_link(self):
        """
        Return:
            base_link_name: str.
        """
        pass

    @abstractmethod
    def get_static_links(self):
        """
        Return:
            base_link_names: list of str.
        """
        pass

    @abstractmethod
    def get_mesh_scale(self):
        """
        Return:
            mesh_scale: float. mesh scale in urdf.
        """
        pass

    @abstractmethod
    def get_contact_field_config(self, digits_cfg):
        """
        Return:
            config. Contact field configuration (e.g., which link to use)
        """
        pass

    def build_additional_filter_func(self, digits_cfg):
        axis_to_idx = {"x": 0, "y": 1, "z": 2}
        rules = digits_cfg.get("additional_filter_rules", [])

        def additional_filter_func(link_name, centroid):
            if not rules:
                return True

            for rule in rules:
                link_name_contains = rule.get("link_name_contains", [])
                if isinstance(link_name_contains, str):
                    link_name_contains = [link_name_contains]

                if link_name_contains and not any(token in link_name for token in link_name_contains):
                    continue

                centroid_min = rule.get("centroid_min", {})
                for axis, threshold in centroid_min.items():
                    if np.any(centroid[:, axis_to_idx[axis]] < threshold):
                        return False

                centroid_max = rule.get("centroid_max", {})
                for axis, threshold in centroid_max.items():
                    if np.any(centroid[:, axis_to_idx[axis]] > threshold):
                        return False

            return True

        return additional_filter_func

    def get_kinematics_tree(self):
        if self.tree is None:
            tree = build_kinematics_tree(
                self.urdf_path,
                active_joint_names=self.get_active_joints(),
            )
            self.tree = tree
        return self.tree

    def get_contact_field(self,digits_cfg):
        config = self.get_contact_field_config(digits_cfg)
        additional_filter_func = self.build_additional_filter_func(digits_cfg)
        tree = build_kinematics_tree(
            self.urdf_path,
            active_joint_names=self.get_active_joints(),
        )
        tree.set_fixed_zero_joints(digits_cfg.get("fixed_zero_joints", digits_cfg.get("disabled_joints", [])))

        contact_field = build_contact_field(
            self.urdf_path, 
            config, 
            additional_filter_func,
            mesh_scale=self.get_mesh_scale(),
            patch_pos_dist_tol=config.get("patch_pos_dist_tol", 0.01),
            patch_rot_dist_tol=config.get("patch_rot_dist_tol", 3.1415 / 5),
        )
        contact_field.generate_contact_field(tree)
        return contact_field
