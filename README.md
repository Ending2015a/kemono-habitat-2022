# kemono-habitat-2022
Kemono agent for habitat challenge 2022


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

## Submit to Remote Machine
```shell
bash scripts/docker-build --remote --submit
# submission command refer to the challenge page
evalai ....
```
