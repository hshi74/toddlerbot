# ToddlerBot

![ToddlerBot](docs/_static/banner.png)

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/rwXqdvVtqh)

|  | Paper | Website | Tweet |
|:--|:-----:|:-------:|:-----:|
| **Locomotion Beyond Feet** (2026) | [arXiv](https://arxiv.org/abs/2601.03607) | [Site](https://locomotion-beyond-feet.github.io/) | [X](https://x.com/taeyang___11/status/2009359173302276391) |
| **ToddlerBot** (2025) | [arXiv](https://arxiv.org/abs/2502.00893) | [Site](https://toddlerbot.github.io/) | [X](https://x.com/HaochenShi74/status/1886599720279400732) |

[Documentation](https://hshi74.github.io/toddlerbot) | [Onshape](https://cad.onshape.com/documents/565bc33af293a651f66e88d2) | [MakerWorld](https://makerworld.com/en/models/1733983)

ToddlerBot is a low-cost, open-source humanoid robot platform designed for scalable policy learning and research in robotics and AI.

This codebase includes low-level control, RL training, DP training, real-world deployment and basically EVERYTHING you need to run ToddlerBot in the real world!

Built entirely in Python, it is **fully pip-installable** (python >= 3.10) for seamless setup and usage!

## News & Updates
- **2026-01-08:** Locomotion Beyond Feet release - multi-skill whole-body locomotion system - [Paper](https://arxiv.org/abs/2601.03607) | [Website](https://locomotion-beyond-feet.github.io/) | [Tweet](https://x.com/taeyang___11/status/2009359173302276391)
- **2025-08-25:** ToddlerBot 2.0 release - see [Changelog](CHANGELOG.md) for details
- **2025-02-03:** ToddlerBot initial release - [Paper](https://arxiv.org/abs/2502.00893) | [Website](https://toddlerbot.github.io/) | [Video](https://youtu.be/A43QxHSgLyM) | [Tweet](https://x.com/HaochenShi74/status/1886599720279400732)

## Locomotion Beyond Feet

We introduce an egocentric multi-skill whole-body locomotion system that enables ToddlerBot to traverse diverse obstacles.

https://github.com/user-attachments/assets/b935f30d-a930-4cff-a9c8-e4cf119ce9c8

<!-- TODO: Replace with actual GitHub video URL after uploading -->

### Getting Started

**1. Create reference motions** using the [Keyframe App](https://github.com/Stanford-TML/robot_keyframe_kit)

**2. Train RL policies** for each skill:
```bash
python toddlerbot/locomotion/train_mjx.py --gin-file <skill_name> --env <skill_name>
```

**3. Collect depth data and train the skill classifier:**

Collect RGB stereo frames. Follow the prompts to select a skill label, press space to start recording, and press space again to pause. Repeat for each skill.
```bash
python toddlerbot/skill_classifier/data/collect_real_world_skill_data.py
```

Process offline to create depth maps from the collected stereo images.
```bash
python toddlerbot/skill_classifier/data/create_depth_data.py
```

Train the classifier on depth images with skill labels.
```bash
python toddlerbot/skill_classifier/training/train.py <data_dir>
```

**4. Run the full system:**

Start the depth estimation server.
```bash
python toddlerbot/skill_classifier/run_foundation_stereo.py
```

Specify the checkpoint path for each trained skill in `POLICY_CONFIGS` in `run_multiple_policy.py` (policy checkpoints can be downloaded from [Google Drive](https://drive.google.com/drive/u/1/folders/1d0UGRc3-3wxMcqrZSlqRXkSnwFSC8Xk-)), then run the multi-skill locomotion system.
```bash
python toddlerbot/policies/run_multiple_policy.py --skill-classifier <classifier_checkpoint>
```

Once the depth estimation server is ready after warm-up and `run_multiple_policy.py` has loaded all policy checkpoints and achieved the standing pose, depth estimates are continuously sent for skill classification, and the robot will perform the appropriate skills autonomously.

## ToddlerBot 2.0
See [Changelog](CHANGELOG.md) for the list of new features and a migration guide from 1.0 to 2.0.

## Setup
Refer to [this page](https://hshi74.github.io/toddlerbot/software/01_setup.html) for instructions to setup.


## Walkthrough

- Checkout `examples` for some scripts to start with. Many of them run on a real-world instance of ToddlerBot.

- The `motion` folder contains carefully crafted keyframe animations designed for ToddlerBot. For example, you can run

    ```
    python toddlerbot/policies/run_policy.py --policy replay --path motion/push_up_2xc.lz4 --vis view
    ```
    (on MacOS)
    ```
    mjpython toddlerbot/policies/run_policy.py --policy replay --path motion/push_up_2xc.lz4 --vis view
    ```
    

    to see the push up motion in MuJoCo. You're very welcome to contribute your keyframe animation to our repository by
    submitting a pull request!

- The `scripts` folder contains some utility bash scripts.

- The `tests` folder have some tests that you can run with 

    ```
    pytest tests/
    ``` 

    to verify our installation.

- The `toddlerbot` folder contains all the source code. You can find a detailed API documentation [here](https://hshi74.github.io/toddlerbot/sections/06_api.html).


## Submitting an Issue
For easier maintenance, we will ONLY monitor GitHub Issues and likely ignore questions from other sources.
We welcome issues related to anything we’ve open-sourced, not just the codebase.

Before submitting an issue, please ensure you have:
- Read the [documentation](https://hshi74.github.io/toddlerbot), including the [Tips and Tricks](https://hshi74.github.io/toddlerbot/sections/05_tips_and_tricks.html) section.
- Checked the comments in the scripts.
- Carefully reviewed the assembly manual.
- Watched the assembly videos.

If we determine that your issue arises from not following these resources, we are unlikely to respond. 
However, if you have found a bug, need support for any open-sourced component, or want to submit a feature request, 
feel free to open an issue.

We truly appreciate your feedback and will do our best to address it!

## Community

Join our <img src="https://cdn.simpleicons.org/discord/5865F2" height="14" alt=""> [Discord](https://discord.gg/rwXqdvVtqh) to chat with the community!

## Contributing  

We welcome contributions from the community! To contribute, just follow the standard practice:
1. Fork the repo  
2. Create a branch (`feature-xyz`)  
3. Commit & push  
4. Submit a Pull Request (PR)  

## Citation
If you use ToddlerBot for published research, please cite:
```bibtex
@misc{yang2026locomotion,
  title = {Locomotion {{Beyond Feet}}},
  author = {Yang, Tae Hoon and Shi, Haochen and Hu, Jiacheng and Zhang, Zhicong and Jiang, Daniel and Wang, Weizhuo and He, Yao and Wu, Zhen and Chen, Yuming and Hou, Yifan and Kennedy, Monroe and Song, Shuran and Liu, C. Karen},
  year = 2026,
  month = jan,
  number = {arXiv:2601.03607},
  eprint = {2601.03607},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2601.03607},
  urldate = {2026-01-08},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Robotics}
}

@article{shi2025toddlerbot,
  title={ToddlerBot: Open-Source ML-Compatible Humanoid Platform for Loco-Manipulation},
  author={Shi, Haochen and Wang, Weizhuo and Song, Shuran and Liu, C. Karen},
  journal={arXiv preprint arXiv:2502.00893},
  year={2025}
}
```

## License  

- The ToddlerBot codebase (including the documentation) is released under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE).

- The ToddlerBot design (Onshape document, STL files, etc.) is released under the [![License: CC BY-NC-SA](https://img.shields.io/badge/License-CC%20BY--NC--SA-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/), which allows you to use and build upon our work non-commercially.
The design of ToddlerBot is provided as-is and without warranty.
