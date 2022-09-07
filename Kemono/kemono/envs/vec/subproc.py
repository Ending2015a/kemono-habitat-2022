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
  'SubprocVecEnv'
]

class CloudpickleWrapper():
  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def __getattr__(self, key: Any) -> Any:
    return self.kwargs.get(key)

  def __getstate__(self) -> Any:
    return cloudpickle.dumps(self.kwargs)

  def __setstate__(self, kwargs: Dict):
    self.kwargs = cloudpickle.loads(kwargs)

# Commands
class CMD(enum.Enum):
  getattr = 1
  setattr = 2
  reset   = 3
  step    = 4
  seed    = 5
  render  = 6
  close   = 7

def _subproc_worker(_p, p, param_wrapper):
  _p.close()
  import setproctitle
  import multiprocessing
  setproctitle.setproctitle(
    multiprocessing.current_process().name
  )
  env = param_wrapper.fn()
  auto_reset = param_wrapper.auto_reset
  try:
    while True:
      try:
        cmd, data = p.recv()
      except EOFError: # the pipe has been closed
        p.close()
        break
      if cmd == CMD.getattr:
        p.send(getattr(env, data[0], None))
      elif cmd == CMD.setattr:
        p.send(setattr(env, data[0], data[1]))
      elif cmd == CMD.reset:
        p.send(env.reset(**data[0]))
      elif cmd == CMD.step:
        obs, rew, done, info = env.step(data[0])
        if auto_reset and done:
          obs = env.reset()
        p.send((obs, rew, done, info))
      elif cmd == CMD.seed:
        p.send(env.seed(data[0]))
      elif cmd == CMD.render:
        p.send(env.render(**data[0]))
      elif cmd == CMD.close:
        p.send(env.close())
        p.close()
        break
      else:
        p.close()
        raise NotImplementedError
  except KeyboardInterrupt:
    p.close()

class SubprocEnvWorker(BaseEnvWorker):
  def __init__(
    self,
    env_fn: Callable,
    worker_id: int,
    auto_reset: bool
  ):
    methods = multiprocessing.get_all_start_methods()
    start_method = 'spawn'
    if 'forkserver' in methods:
      start_method = 'forkserver'
    ctx = multiprocessing.get_context(start_method)
    self.p, _p = ctx.Pipe()
    args = (
      self.p, _p, CloudpickleWrapper(fn=env_fn, auto_reset=auto_reset)
    )
    self.worker_id = worker_id
    self.process = ctx.Process(
      target = _subproc_worker,
      args = args,
      name = f'SubprocEnvWorker-{worker_id}',
      daemon = True
    )
    self.process.start()
    self._waiting_cmd = None
    _p.close()
    super().__init__(env_fn, auto_reset)
    
  def getattr(self, attrname: str) -> Any:
    return self._exec(CMD.getattr, attrname)

  def setattr(self, attrname: str, value: Any) -> Any:
    return self._exec(CMD.setattr, attrname, value)

  def reset(self, **kwargs) -> Any:
    return self._exec(CMD.reset, kwargs)

  def step_async(self, act: Any):
    self._exec(CMD.step, act, block=False)

  def step_wait(self, timeout: Optional[int]=None) -> Any:
    return self._wait(CMD.step, timeout=timeout)

  def seed(self, seed: int) -> Any:
    super().seed(seed)
    return self._exec(CMD.seed, seed)

  def render(self, **kwargs) -> Any:
    return self._exec(CMD.render, kwargs)

  def close_async(self):
    self._exec(CMD.close, block=False)

  def close_wait(self) -> Any:
    return self._wait(CMD.close, timeout=1)

  def _exec(
    self,
    cmd: CMD,
    *args,
    block: bool = True,
    timeout: Optional[int] = None
  ):
    #TODO: find a more reliable way
    if self._waiting_cmd and cmd != CMD.close:
      raise RuntimeError(f"Another command {cmd} was sent when "
        f"waiting for the reply {self._waiting_cmd}.")
    self.p.send([cmd, args])
    self._waiting_cmd = cmd
    if block:
      return self._wait(cmd, timeout=timeout)

  def _wait(self, cmd: CMD, timeout: Optional[int] = None):
    if self._waiting_cmd != cmd:
      raise RuntimeError(f"Waiting for command {cmd} but another command "
        f"{self._waiting_cmd} is executing.")
    res = None
    if self.p.poll(timeout):
      res = self.p.recv()
    self._waiting_cmd = None #unmarked
    return res


class SubprocVecEnv(BaseVecEnv):
  def __init__(
    self,
    env_fns: List[Callable],
    **kwargs
  ):
    kwargs.pop('worker_class', None)
    super().__init__(env_fns, SubprocEnvWorker, **kwargs)