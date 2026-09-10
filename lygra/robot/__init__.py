# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.robot.sharpa import Sharpa
from lygra.robot.wuji import Wuji


ROBOT_TYPES = {
    "sharpa": Sharpa,
    "wuji": Wuji,
}


def build_robot(robot_name="sharpa", urdf_path=None):
    try:
        robot_type = ROBOT_TYPES[robot_name]
    except KeyError as exc:
        supported = ", ".join(sorted(ROBOT_TYPES))
        raise ValueError(
            f"Unsupported robot '{robot_name}'. Supported robots: {supported}"
        ) from exc
    return robot_type(urdf_path=urdf_path)
