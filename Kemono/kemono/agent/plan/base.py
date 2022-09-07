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

@dataclass
class Plan:
  plan_map: Optional[np.ndarray] = None
  plan_act: Optional[int] = None
  center_pos: Optional[np.ndarray] = None

class BasePlanner:
  def set_agent(self, agent):
    raise NotImplementedError

  def reset(self) -> Plan:
    raise NotImplementedError

  def act(self, observations) -> Plan:
    raise NotImplementedError
