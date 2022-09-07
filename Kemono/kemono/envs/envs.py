# --- built in ---
import copy
from typing import (
  Optional
)
# --- 3rd party ---
import habitat
import numpy as np
import gym
from habitat.datasets import make_dataset
import cv2
# --- my module ---

__all__ = [
  'DummyEnv',
  'Emv'
]

class HabitatEnvSpec:
  version: int = 0
  id: str = "HabitatEnv-v{}"
  def __init__(self, version: int=0):
    self.version = 0
    self.id = self.id.format(version)


class _BaseEnv(gym.Env):
  metadata = {'render.modes': ['rgb_array', 'human']}
  reward_range = {-float('inf'), float('inf')}
  spec = HabitatEnvSpec()

  def __init__(
    self,
    config: habitat.Config,
    version: int = 0,
    no_pitch: bool = False
  ):
    self._no_pitch = no_pitch
    self.config = config
    self.spec = HabitatEnvSpec(version=version)

    self.observation_space = self.make_observation_space()
    self.action_space = self.make_action_space()
    self._cached_obs = None

  def make_observation_space(self):
    rgb_config = self.config.SIMULATOR.RGB_SENSOR
    depth_config = self.config.SIMULATOR.DEPTH_SENSOR

    dim = 1 if self._no_pitch else 2
    compass_space = gym.spaces.Box(
      high = np.pi,
      low = -np.pi,
      shape = (dim,),
      dtype = np.float32
    )
    objectgoal_space = gym.spaces.Discrete(6)
    rgb_space = gym.spaces.Box(
      high = 255,
      low = 0,
      shape = (rgb_config.HEIGHT, rgb_config.WIDTH, 3),
      dtype = np.uint8
    )
    depth_space = gym.spaces.Box(
      high = 1.,
      low = 0.,
      shape = (depth_config.HEIGHT, depth_config.WIDTH, 1),
      dtype = np.float32
    )
    return gym.spaces.Dict({
      'rgb': rgb_space,
      'depth': depth_space,
      'compass': compass_space,
      'objectgoal': objectgoal_space
    })

  def make_action_space(self):
    # ///-1 = STOP
    # 0 = MOVE_FORWARD
    # 1 = TURN_LEFT
    # 2 = TURN_RIGHT
    return gym.spaces.Discrete(3)

  def get_observation(self, obs):
    if not self._no_pitch:
      compass = obs['compass']
      pitch = np.radians(0.0)
      obs['compass'] = np.concatenate((compass, [pitch]), axis=0)
    return obs

  def render(self, mode='human'):
    if self._cached_obs is None:
      return
    scene = self.render_scene()
    if mode == 'rgb_array':
      return scene
    else:
      cv2.imshow('scene', scene[...,::-1])
  
  def render_scene(self):
    assert self._cached_obs is not None
    obs = self._cached_obs
    rgb = obs['rgb']
    depth = obs['depth']
    depth = (np.concatenate((depth,)*3, axis=-1) * 255.0).astype(np.uint8)
    scene = np.concatenate((rgb, depth), axis=1)
    return scene

  def seed(self, seed: Optional[int]=None) -> None:
    pass

  @classmethod
  def make(cls, id, *args, **kwargs):
    env_name, version = id.split('-v')
    version = int(version)
    return cls(*args, **kwargs, version=version)


class DummyEnv(_BaseEnv):
  def __init__(
    self,
    config: habitat.Config,
    version: int = 0
  ):
    super().__init__(config, version, no_pitch=False)
  
  def set_observation(self, obs):
    self._cached_obs = copy.deepcopy(obs)

  def reset(self, *args, **kwargs):
    return self.get_observation(self._cached_obs)

  def step(self, act, **kwargs):
    obs = self.get_observation(self._cached_obs)
    # dummy infos
    rew = 0
    done = False
    info = {}
    return (obs, rew, done, info)


class Env(_BaseEnv):
  def __init__(
    self,
    config: habitat.Config,
    dataset = None,
    version: int = 0
  ):
    super().__init__(config, version, no_pitch=True)

    if dataset is None:
      dataset = make_dataset(
        id_dataset = self.config.DATASET.TYPE,
        config = self.config.DATASET
      )
    self.dataset = dataset
    self._env = habitat.Env(self.config, dataset=self.dataset)

  @property
  def habitat_env(self):
    return self._env

  def reset(self, *args, **kwargs):
    return self.get_observation(self._env.reset(*args, **kwargs))

  def step(self, act, **kwargs):
    # 0 = STOP
    # 1 = MOVE_FORWARD
    # 2 = TURN_LEFT
    # 3 = TURN_RIGHT
    obs = self._env.step(act, **kwargs)
    obs = self.get_observation(obs)
    rew = 0
    done = self._env.episode_over
    info = self._env.get_metrics()
    return (obs, rew, done, info)

  def seed(self, seed: Optional[int]=None) -> None:
    self._env.seed(seed)
