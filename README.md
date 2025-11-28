# HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos

<div align="center">
<a href="https://hdmi-humanoid.github.io/">
  <img alt="Website" src="https://img.shields.io/badge/Website-Visit-blue?style=flat&logo=google-chrome"/>
</a>

<a href="https://www.youtube.com/watch?v=GvIBzM7ieaA&list=PL0WMh2z6WXob0roqIb-AG6w7nQpCHyR0Z&index=12">
  <img alt="Video" src="https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=youtube"/>
</a>

<a href="https://arxiv.org/pdf/2509.16757">
  <img alt="Arxiv" src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?style=flat&logo=arxiv"/>
</a>

<a href="https://github.com/LeCAR-Lab/HDMI/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/LeCAR-Lab/HDMI?style=social"/>
</a>


</div>

HDMI is a novel framework that enables humanoid robots to acquire diverse whole-body interaction skills directly from monocular RGB videos of human demonstrations.

This repository contains the official training code of **HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos**.


## TODO
- [x] Release hdmi training code 
- [x] hoi motion datasets
- [x] Release pretrained models
- [x] Release sim2real code


## 🚀 Quick Start

```bash
# setup conda environment
conda create -n hdmi python=3.11 -y
conda activate hdmi

# install isaacsim
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
isaacsim # test isaacsim

# install isaaclab
cd ..
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.2.0
./isaaclab.sh -i none

# install hdmi
cd ..
git clone https://github.com/LeCAR-Lab/HDMI
cd HDMI
pip install -e .
```

## Data Preparation

### Desired Data Format
The training scripts load motion data from a `motion.npz` file (see `active_adaptation/utils/motion.py`). The archive stores a Python dict with the following keys and shapes (from [issue #2](https://github.com/LeCAR-Lab/HDMI/issues/2)):
- Body states: `pos`, `quat`, `lin_vel`, `ang_vel` with shape `[T, B, 3/4]`.
- Joint states: `pos`, `vel` with shape `[T, J]`.

`T` is the number of time steps, `B` is the number of bodies (including any objects you append), and `J` is the number of joints. The exact body and joint ordering is defined in the `meta.json` that sits alongside each `motion.npz`.

### Processing Steps
If you need to turn HOI/video data into this format (per the conversion outline in [issue #2](https://github.com/LeCAR-Lab/HDMI/issues/2)), the workflow is:
1) Convert human motion to robot motion using GVHMR → GMR/LocoMujoco to get robot body/joint states.
2) Extract the object trajectory (with some heuristic rules).
3) Append the object name to the body list in `meta.json`, then concatenate the object body states (`pos`, `quat`, `lin_vel`, `ang_vel`) to the robot body states so the resulting shapes will be `[T, B_robot + B_object, 3/4]`.

### Verify Your Data
You can add a `+task.command.record_motion=true` flag to any scripts to visualize the motion in Isaac Sim.

```bash
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase +task.command.replay_motion=true
```

You can also use the following script to visualize a `motion.npz` file in mujoco:

```bash
# one terminal
python scripts/vis/mujoco_mocap_viewer.py
# another terminal
python scripts/vis/motion_data_publisher.py <path-to-motion-folder>
```

## Train and Evaluate

Teacher policy 
```bash
# train policy
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase
# evaluate policy
python scripts/play.py algo=ppo_roa_train task=G1/hdmi/move_suitcase checkpoint_path=run:<wandb-run-path>
```

Student policy
```bash
# train policy
python scripts/train.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<teacher_wandb-run-path>
# evaluate policy
python scripts/play.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<student_wandb-run-path>
```

To export trained policies add a `export_policy=true` flag to the play script.


## Sim2Real

Please see [github.com/EGalahad/sim2real](https://github.com/EGalahad/sim2real) for details.

## Citation

If you find our work useful for your research, please consider cite us:

```
@misc{weng2025hdmilearninginteractivehumanoid,
      title={HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos}, 
      author={Haoyang Weng and Yitang Li and Nikhil Sobanbabu and Zihan Wang and Zhengyi Luo and Tairan He and Deva Ramanan and Guanya Shi},
      year={2025},
      eprint={2509.16757},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.16757}, 
}
```
