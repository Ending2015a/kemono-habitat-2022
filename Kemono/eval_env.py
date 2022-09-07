# --- built in ---
import os
import json
import random
import argparse
# --- 3rd party ---
import habitat
import numpy as np
import torch
from omegaconf import OmegaConf
# --- my module ---
from arguments import get_args
from info_recorder import InfoRecorder
from kemono.agent import KemonoAgent
from kemono.envs import Env


def main():
  a = get_args()
  # load yaml configs
  conf = OmegaConf.merge(
    OmegaConf.load(a.config), # yaml file
    OmegaConf.from_dotlist(a.dot_list) # command line
  )
  # load habitat configs
  habitat_config_path = conf.habitat_config
  habitat_config = habitat.get_config(habitat_config_path)
  # specify which gpu to use
  torch.cuda.set_device(a.gpu)
  print(f'Device count: {torch.cuda.device_count()}, '
    f'use gpu: {a.gpu}, current device: {torch.cuda.current_device()}')
  # apply dataset config
  habitat_config.defrost()
  # check if this works properly
  habitat_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = a.gpu
  habitat_config.freeze()

  # save config to log
  os.makedirs(conf.log_path, exist_ok=True)
  filename = os.path.join(conf.log_path, 'config.yaml')
  OmegaConf.save(config=conf, f=filename, resolve=True)

  # create agent
  agent = KemonoAgent(
    conf = conf,
    habitat_config = habitat_config
  )
  # create env
  env = Env.make('habitatEnv-v0', habitat_config)
  recorder = InfoRecorder(conf.log_path)

  # run evaluations
  for i in range(conf.eval_num_episodes):
    agent.reset()
    obs = env.reset()
    while True:
      act = agent.act(obs)
      obs, rew, done, info = env.step(act['action'])
      if done:
        break
    info['episode'] = i + 1
    recorder.add(info)
  # print averaged results
  print(recorder.avg_info)

if __name__ == "__main__":
  main()