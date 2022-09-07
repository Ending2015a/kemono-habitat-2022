# --- built in ---
import abc
import enum
import multiprocessing
from typing import (
  Any,
  Callable,
  List,
  Optional,
  Tuple,
  Union,
  Dict
)
# --- 3rd party ---
import gym
import numpy as np
import cloudpickle
# --- my module ---
from rlchemy.lib import utils as rl_utils
from kemono.envs.vec.base import BaseVecEnv, BaseEnvWorker

__all__ = [
  'DummyVecEnv',
  'VecEnv'
]

class EnvWorker(BaseEnvWorker):
  def __init__(
    self,
    env_fn: Callable,
    worker_id: int,
    auto_reset: bool
  ):
    self.env = env_fn()
    self.worker_id = worker_id
    self._res = None
    super().__init__(env_fn, auto_reset)
  
  def getattr(self, attrname: str) -> Any:
    return getattr(self.env, attrname)

  def setattr(self, attrname: str, value: Any) -> Any:
    return setattr(self.env, attrname, value)

  def reset(self, **kwargs) -> Any:
    return self.env.reset(**kwargs)

  def step_async(self, act: Any):
    obs, rew, done, info = self.env.step(act)
    if self._auto_reset and done:
      obs = self.env.reset()
    self._res = (obs, rew, done, info)

  def step_wait(self) -> Any:
    return self._res

  def seed(self, seed: int) -> Any:
    super().seed(seed)
    return self.env.seed(seed)

  def render(self, **kwargs) -> Any:
    return self.env.render(**kwargs)

  def close_async(self):
    self.env.close()

  def close_wait(self):
    pass

class DummyVecEnv(BaseVecEnv):
  def __init__(
    self,
    env_fns: List[Callable],
    **kwargs
  ):
    kwargs.pop('worker_class', None)
    super().__init__(env_fns, EnvWorker, **kwargs)

class VecEnv(BaseVecEnv):
  def __init__(
    self,
    envs: List[gym.Env],
    **kwargs
  ):
    kwargs.pop('worker_class', None)
    env_fns = [lambda i=j: envs[i] for j in range(len(envs))]
    super().__init__(env_fns, EnvWorker, **kwargs)