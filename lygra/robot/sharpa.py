# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from lygra.robot.base import RobotInterface

class Sharpa(RobotInterface):
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
        return '../assets/hands/sharpa/left_grasp.urdf'

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
        l = ['left_index_MCP_FE', 'left_index_MCP_AA', 'left_index_PIP', 'left_index_DIP', 'left_middle_MCP_FE', 'left_middle_MCP_AA', 'left_middle_PIP', 'left_middle_DIP', 'left_pinky_CMC', 'left_pinky_MCP_FE', 'left_pinky_MCP_AA', 'left_pinky_PIP', 'left_pinky_DIP', 'left_ring_MCP_FE', 'left_ring_MCP_AA', 'left_ring_PIP', 'left_ring_DIP', 'left_thumb_CMC_FE', 'left_thumb_CMC_AA', 'left_thumb_MCP_FE', 'left_thumb_MCP_AA', 'left_thumb_IP']
        return l

    def get_base_link(self):
        return "left_hand_C_MC"

    def get_static_links(self):
        return ["left_hand_C_MC"]

    def get_mesh_scale(self):
        """
            Your robot mesh might be rescaled in your URDF, specify it here.
            (will be removed in the future.)
        """
        return 1.0
