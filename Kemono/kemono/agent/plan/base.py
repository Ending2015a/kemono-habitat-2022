# --- built in ---
from dataclasses import dataclass
from typing import Optional, List
# --- 3rd party ---
import habitat
import numpy as np
# --- my module ---

__all__ = [
  'Plan',
  'BasePlanner'
]


def merge_plans(*plans, strategy='greedy'):
  if len(plans) == 0:
    return Plan()

  final = Plan()
  for plan in plans:
    # fill in plan map
    if final.plan_map is None:
      final.plan_map = plan.plan_map
      final.center_pos = plan.center_pos
    # fill in plan act
    if final.plan_act is None:
      final.plan_act = plan.plan_act

  return final

@dataclass
class Plan:
  plan_map: Optional[np.ndarray] = None
  plan_act: Optional[int] = None
  center_pos: Optional[np.ndarray] = None

  def merge(self, *plans, strategy='greedy'):
    return merge_plans(self, *plans, strategy=strategy)

class BasePlanner:
  def set_agent(self, agent):
    raise NotImplementedError

  def reset(self) -> Plan:
    raise NotImplementedError

  def act(self, observations) -> Plan:
    raise NotImplementedError
