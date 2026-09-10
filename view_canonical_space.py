import argparse

from lygra.robot import build_robot
from lygra.utils.robot_visualizer import RobotVisualizer
from lygra.utils.vis_utils import get_box_lineset_visual


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="sharpa")
    parser.add_argument("--urdf-path")
    args = parser.parse_args()

    robot = build_robot(args.robot, urdf_path=args.urdf_path)

    box_min, box_max = robot.get_canonical_space()

    print("min", box_min)
    print("max", box_max)

    robot_tree = robot.get_kinematics_tree()
    
    lower, upper = robot_tree.get_active_joint_limit()
    q = (lower + upper) / 2

    box = get_box_lineset_visual(box_min, box_max)

    viewer = RobotVisualizer(robot)
    robot_mesh = viewer.get_mesh_fk(q, visual=False)
    viewer.show(robot_mesh + [box])
