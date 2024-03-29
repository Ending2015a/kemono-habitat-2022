# kemono-habitat-2022

Kemono agent for habitat challenge 2022


Kemono agent is a *pure-rule-based* object-goal-navigation agent. It can achieve SPL **0.3099** and SUCCESS **0.58** on the [test-standard phase of habitat challenge 2022](https://eval.ai/web/challenges/challenge-page/1615/overview).

<img src="https://github.com/Ending2015a/kemono-habitat-2022/blob/master/assets/system.png" width="50%">




https://user-images.githubusercontent.com/18180004/190056280-6096fdce-2d84-4e7b-a33c-710fb4c3d07c.mp4




version: `v0.0.2`

It uses the following packages:
* [Ending2015a/dungeon_maps](https://github.com/Ending2015a/dungeon_maps): A highly accurate semantic top-down mapper
* [Ending2015a/rlchemy](https://github.com/Ending2015a/rlchemy): Reinforcement Learning package

## Performance

| Version | SPL | SUCCESS |
|-|-|-|
| v0.0.2  | 0.3136 | 0.571 |
| v0.0.1  | 0.3099 | 0.58  |

## Evaluate on Local Machine

1. [Download the pretrained RedNet checkpoint here](https://drive.google.com/file/d/1n7_c352ftcTHR-USYhnfQSbvjX-5i-8Y/view?usp=sharing), place it at `Kemono/weights/*.ckpt`
2. Build the docker by the following command
```shell
bash scripts/docker-build
```
3. Setup the following variables:
```shell
export HABITAT_DATA_DIR="/path/to/your/habitat-challenge-data"
export HABITAT_LOG_DIR="/path/to/your/log/dir"
```
4. Start a container and run the evaluation process:
    * Non-interactive mode
      ```shell
      bash scripts/docker-start
      ```
    * Interactive mode (bash)
      ```shell
      bash scripts/docker-start --bash
      . activate habitat
      bash ./scripts/evaluation.sh
      ```

## Submit to EvalAI
```shell
bash scripts/docker-build --remote --submit
# submission command refer to the challenge page
evalai ....
```
