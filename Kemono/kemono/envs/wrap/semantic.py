# --- built in ---
import enum
from typing import Any, Dict, List, Union
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
import rlchemy
from rlchemy import registry
# --- my module ---
from kemono.semantic import SemanticMapping
from kemono.semantic.task import SemanticPredictor

class Predictor():
  def __init__(
    self,
    semap: SemanticMapping,
    goal_scale: float=0.5,
    **predictor_kwargs
  ):
    self.semap = semap
    self.goal_scale = goal_scale
    self.predictor = SemanticPredictor(
      **predictor_kwargs
    )

  def predict(self, obs):
    rgb = obs['rgb'].astype(np.uint8) # (h, w, 3)
    depth = obs['depth'].astype(np.float32) # (h, w, 1)
    seg_raw = self.predictor.predict(rgb, depth)
    goal_id = self.semap.get_goal_category_id(obs['objectgoal'])
    seg_raw[goal_id] *= self.goal_scale
    return seg_raw


class SemanticWrapper(gym.Wrapper):
  seg_key = 'seg'
  seg_raw_key = 'seg_raw'
  seg_color_key = 'seg_color'
  def __init__(
    self,
    env: habitat.RLEnv,
    predictor_config: Dict[str, Any],
    goal_mapping: Dict[int, str],
    colorized: bool = True
  ):
    super().__init__(env=env)
    self._cached_obs = None
    self._cached_seg_raw = None
    self.colorized = colorized
    self.semap = SemanticMapping(goal_mapping)
    self.predictor = Predictor(
      semap=self.semap, **predictor_config)

    self.observation_space = self.make_observation_space()
    self._setup_interact = False

  @property
  def semantic_mapping(self):
    return self.semap

  def step(self, *args, **kwargs):
    obs, rew, done, info = self.env.step(*args, **kwargs)
    obs = self.get_observation(obs)
    info = self.get_info(obs, info)
    return obs, rew, done, info

  def reset(self, *args, **kwargs):
    obs = self.env.reset(*args, **kwargs)
    obs = self.get_observation(obs)
    return obs

  def make_observation_space(self):
    if self.predictor is None:
      return self.env.observation_space
    if self.env.observation_space is None:
      return
    width = 640
    height = 480
    obs_space = self.observation_space
    new_obs_spaces = {key: obs_space[key] for key in obs_space}
    # raw semantic id spaces (category ids)
    seg_space = gym.spaces.Box(
      low = 0,
      high = 39,
      shape = (height, width),
      dtype = np.int32
    )
    new_obs_spaces[self.seg_key] = seg_space
    # raw semantic predictions (logits)
    seg_raw_space = gym.spaces.Box(
      low = -float('inf'),
      high = float('inf'),
      shape = (40, height, width),
      dtype = np.float32
    )
    new_obs_spaces[self.seg_raw_key] = seg_raw_space
    # colorized semantic spaces (RGB image)
    if self.colorized:
      seg_color_space = gym.spaces.Box(
        low = 0,
        high = 255,
        shape = (height, width, 3),
        dtype = np.uint8
      )
      new_obs_spaces[self.seg_color_key] = seg_color_space
    # create new Dict spaces
    new_obs_space = gym.spaces.Dict(new_obs_spaces)
    return new_obs_space

  def get_observation(self, obs):
    # predict category id map (40, h, w)
    seg_raw = self.predictor.predict(obs)
    # seg_raw = self.predictor.predict(obs)
    obs[self.seg_raw_key] = seg_raw
    # predict category id map
    seg = np.argmax(seg_raw, axis=0) # (h, w)
    obs[self.seg_key] = seg
    # colorize segmentation map
    seg_color = self.semap.colorize_categorical_map(seg, rgb=True)
    obs[self.seg_color_key] = seg_color
    self._cached_obs = obs
    return obs

  def get_info(self, obs, info):
    info['goal'] = {
      'id': self.semap.get_goal_category_id(obs['objectgoal']),
      'name': self.semap.get_goal_category_name(obs['objectgoal'])
    }
    return info
  
  def on_dubclick_probe_info(self, event, x, y, flags, param):
    seg_raw = self._cached_obs[self.seg_raw_key]
    seg_exp = np.exp(seg_raw)
    seg_prob = seg_exp / np.sum(seg_exp, axis=0)
    if event == cv2.EVENT_LBUTTONDOWN:
      np.set_printoptions(precision=5, suppress=True)
      print(f"On click (x, y) = ({x}, {y})")
      print(np.reshape(seg_prob[:, y, x], (8, 5))) # 8*5 = 40 classes

  def render(self, mode='human'):
    seg_color = self._cached_obs[self.seg_color_key]
    if mode == 'interact':
      window_name = 'SemanticWrapper (interactive)'
      if not self._setup_interact:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_dubclick_probe_info)
        self._setup_interact = True
      cv2.imshow(window_name, seg_color[...,::-1])
    elif mode == 'human':
      cv2.imshow('semantic', seg_color[...,::-1])
    return seg_color

