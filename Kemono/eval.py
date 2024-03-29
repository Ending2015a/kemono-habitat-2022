# --- built in ---
import os
import argparse
import random
# --- 3rd party ---
import habitat
import torch
import numpy as np
from omegaconf import OmegaConf
# --- my module ---
from arguments import get_args
from kemono.agent import KemonoAgent

def main():
  a = get_args()
  # load yaml configs
  conf = OmegaConf.merge(
    OmegaConf.load(a.config), # yaml file
    OmegaConf.from_dotlist(a.dot_list) # command line
  )
  # load habitat configs
  habitat_config_path = os.environ['CHALLENGE_CONFIG_FILE']
  habitat_config = habitat.get_config(habitat_config_path)
  # create agent
  agent = KemonoAgent(
    conf = conf,
    habitat_config = habitat_config
  )
  # submit challenge
  if a.evaluation == "local":
    challenge = habitat.Challenge(eval_remote=False)
  else:
    challenge = habitat.Challenge(eval_remote=True)
  challenge.submit(agent)

if __name__ == "__main__":
  main()