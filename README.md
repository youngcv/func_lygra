## Installation
Please refer to [lightning grasp](<https://github.com/zhaohengyin/lightning-grasp>) for installation instructions.

## Run
Every run needs:

- `INPUT_DIR`: directory containing object instance folders, each with `labeled.obj`
- `OUTPUT_DIR`: directory where generated grasps will be written

Batch multiple instances across GPUs:
```
INPUT_DIR=../assets/objects/knife_30 \
OUTPUT_DIR=../caches/initial_grasp \
GPUS=0,1 bash scripts/batch.sh knife
```

Run one specific instance with visualization:
```
INPUT_DIR=../assets/objects/knife_30 \
OUTPUT_DIR=../caches/initial_grasp \
VISUALIZE=True \
bash scripts/batch.sh knife 000
```

Input files are read from `<input>/<instance_id>/labeled.obj`; outputs are saved
to `<output>/<robot>/<input-dir-name>/<instance_id>/`.

Ensure your object mesh is scaled to meter units before processing.

Select the another hand with a Hydra override:
```
INPUT_DIR=../assets/objects/knife_30 \
OUTPUT_DIR=../caches/initial_grasp \
bash scripts/knife.sh robot=wuji
```

Results are saved under the selected hand name, for example
`<output>/wuji/knife_real/000`.

Generation parameters are stored with each object in `configs/object/<name>.yaml`:

## Setup Your Model
Each hand needs:

1. A `RobotInterface` implementation under `lygra/robot/`.
2. A registration entry in `lygra/robot/__init__.py`.
3. A matching `digits.<robot_name>` section in every supported object YAML.


Object sampling settings remain directly under `parts`; hand-specific object
contact rules live under `parts.robot_rules.<robot_name>`.
