# --- built in ---
from collections import deque
# --- 3rd party ---
import habitat
import numpy as np
import rlchemy
from rlchemy import registry
# --- my module ---
from kemono.agent.plan.base import Plan, BasePlanner

@registry.register.planner('start_up')
class StartUpPlanner(BasePlanner):
  def __init__(
    self,
    actions
  ):
    super().__init__()
    self.actions = actions

    self.action_deque = None

  def set_agent(self, agent):
    self.agent = agent

  def reset(self):
    self.action_deque = deque(self.actions)
  
  def act(self, observations):
    if len(self.action_deque) > 0:
      act = self.action_deque.popleft()
    else:
      act = None
    return Plan(plan_act=act)
