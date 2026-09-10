# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from lygra.robot.base import RobotInterface

class Wuji(RobotInterface):
    def get_canonical_space(self):
        """
            Canonical Space for placing objects (we randomly select an object surface point and drag it into this box)
        """
        box_min = np.array([0.06, -0.03, 0.12], dtype=np.float32)
        box_max = np.array([0.11, 0.03, 0.18], dtype=np.float32)
        return box_min, box_max 

    def get_default_urdf_path(self):
        """
            Default path to your robot URDF
        """
        return '../assets/hands/wuji/right.urdf'

    def get_contact_field_config(self, digits_cfg):
        """
            Specify which links should make contact:
            - Static Links: e.g. palms.
            - Movable Links: e.g. fingers.

            For movable links, you can restrict contact normal directions using the format below.
        """
        config = {
            "type": "v1",
            "movable_link": {},
            "static_link": {},
            "specific_link": []
        }
        config["specific_link"] = digits_cfg['specific_links']
        for link in digits_cfg['movable_links']:
            ll = []
            for vec, ang in digits_cfg["movable_disabled_normal"]:
                ll.append((np.array(vec), np.pi * ang))
            config["movable_link"][link] = {
                "disabled_normal": ll
            }
            if link in digits_cfg["override_disabled_normal"].keys():
                ll = []
                for vec, ang in digits_cfg["override_disabled_normal"][link]:
                    ll.append((np.array(vec), np.pi * ang))
                config["movable_link"][link] = {
                    "disabled_normal": ll
                }
        for link in digits_cfg['static_links']:
            vec = digits_cfg["static_allowed_normal"][0]
            ang = digits_cfg["static_allowed_normal"][1]
            config["static_link"][link] = {
                "allowed_normal": [(np.array(vec), np.pi * ang)]
            }
        return config
    def get_active_joints(self):
        """
            Specify the active joints (dofs).
            Our system will return active joint values in this order.
        """
        l = ['finger1_joint1', 'finger1_joint2', 'finger1_joint3', 'finger1_joint4', 'finger2_joint1', 'finger2_joint2', 'finger2_joint3', 'finger2_joint4', 'finger3_joint1', 'finger3_joint2', 'finger3_joint3', 'finger3_joint4', 'finger4_joint1', 'finger4_joint2', 'finger4_joint3', 'finger4_joint4', 'finger5_joint1', 'finger5_joint2', 'finger5_joint3', 'finger5_joint4']
        return l

    def get_base_link(self):
        return "palm_link"

    def get_static_links(self):
        return ["palm_link"]

    def get_mesh_scale(self):
        """
            Your robot mesh might be rescaled in your URDF, specify it here.
            (will be removed in the future.)
        """
        return 1.0
