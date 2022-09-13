# kemono-habitat-2022

Kemono agent for habitat challenge 2022


Kemono agent is a *pure-rule-based* object-goal-navigation agent. It can achieve SPL **0.3099** and SUCCESS **0.58** on the [test-standard phase of habitat challenge 2022](https://eval.ai/web/challenges/challenge-page/1615/overview).

<img src="https://github.com/Ending2015a/kemono-habitat-2022/blob/master/assets/system.png" width="50%">



https://user-images.githubusercontent.com/18180004/189946415-393b9957-43a7-457f-a2f3-bf5c6f8c4ce3.mp4



version: `v0.0.1`

It uses the following packages:
* [Ending2015a/dungeon_maps](https://github.com/Ending2015a/dungeon_maps): A highly accurate semantic top-down mapper
* [Ending2015a/rlchemy](https://github.com/Ending2015a/rlchemy): Reinforcement Learning package

## Evaluate on Local Machine

First build the docker by the following command
```shell
bash scripts/docker-build
```

Then, you have to set the following variables before you run the evaluation process:
```shell
export HABITAT_DATA_DIR="/path/to/your/habitat-challenge-data"
export HABITAT_LOG_DIR="/path/to/your/log/dir"
```

* Run in a non-interactive container
```shell
bash scripts/docker-start
```
* Run in an interactive container
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
