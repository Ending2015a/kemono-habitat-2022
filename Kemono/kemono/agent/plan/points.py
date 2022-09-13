# --- built in ---
# --- 3rd party ---
import habitat
import numpy as np
import rlchemy
from rlchemy import registry
# --- my module ---
from kemono.agent.plan.base import Plan, BasePlanner

@registry.register.planner('points', default=True)
class PointsPlanner(BasePlanner):
  def __init__(
    self,
    map_size: int = 240,
    radius: float = 0.56,
    angle: float = 45,
    n_points: int = 4,
    local_steps: int = 25
  ):
    super().__init__()
    self.map_size = map_size
    self.radius = radius
    self.angle = angle
    self.n_points = n_points
    self.local_steps = local_steps

    self.plan_points = self.make_plan_space()
    self.agent = None

    self.center_pos = None
    self.curr_plan_id = 0
    self.curr_step = 0
    self.need_update_local = True

  def set_agent(self, agent):
    self.agent = agent
  
  def reset(self):
    self.center_pos = None
    self.curr_plan_id = 0
    self.curr_step = 0
    self.need_update_local = True

  def act(self, observations):
    if self.agent.need_replan:
      self.set_next_plan()

    if self.curr_step == self.local_steps:
      self.curr_step = 0
      self.need_update_local = True

    if self.need_update_local:
      center_pos = self.agent.controller.curr_pose[:2]
      self.center_pos = center_pos
      self.need_update_local = False

    plan_map = np.zeros((self.map_size, self.map_size), dtype=bool)
    plan_point = self.plan_points[self.curr_plan_id] * self.map_size
    # set long-term goal
    plan_map[int(plan_point[0]), int(plan_point[1])] = True
    center_pos = self.center_pos

    # increment step
    self.curr_step = self.curr_step + 1

    return Plan(plan_map=plan_map, center_pos=center_pos)


  def set_next_plan(self):
    self.curr_plan_id = (self.curr_plan_id + 1) % self.n_points

  def make_plan_space(self):
    radius = self.radius
    angle = self.angle
    n_points = self.n_points

    degs = np.linspace(angle, angle-360, n_points, False)
    rads = degs / 180 * np.pi
    points = np.stack((np.cos(rads), np.sin(rads)), axis=-1)
    points = points * radius + 0.5
    points = np.clip(points, 0.0, 1.0)
    return points

