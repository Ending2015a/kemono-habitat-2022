# --- built in ---
import os
from typing import List, Optional
# --- 3rd party ---
import habitat
import cv2
import numpy as np
import skfmm
import skimage
import matplotlib.pyplot as plt
# --- my module ---
from kemono.utils import image as image_utils

def get_mask(sx, sy, step_size):
  size = int(step_size) * 2 + 1
  mask = np.zeros((size, size))
  for i in range(size):
    for j in range(size):
      ii = ((i + 0.5) - (size // 2 + sx)) ** 2
      jj = ((j + 0.5) - (size // 2 + sy)) ** 2
      if ((ii + jj <= step_size ** 2) and
          (ii + jj > (step_size-1) ** 2)):
        mask[i, j] = 1
  mask[size // 2, size // 2] = 1
  return mask

def get_dist(sx, sy, step_size):
  size = int(step_size) * 2 + 1
  mask = np.zeros((size, size)) + 1e-10
  for i in range(size):
    for j in range(size):
      ii = ((i + 0.5) - (size // 2 + sx)) ** 2
      jj = ((j + 0.5) - (size // 2 + sy)) ** 2
      if (ii + jj <= step_size ** 2):
        mask[i, j] = max(5, (ii+jj) ** 0.5)
  return mask

class FMMPlanner():
  def __init__(
    self,
    traversible: np.ndarray,
    class_costs: List[float] = [100000., 1.],
    collision_cost: float = 1000.,
    step_size: float = 0.25,
    stop_distance: float = 0.25,
    map_res: float = 0.05,
  ):
    """Fast-marching short-term goal planner

    Args:
      traversible (np.ndarray): traversible map, np.int64 or np.bool
      class_costs (List[float], optional): traversal cost of each class.
        Defaults to [100000., 1.].
      collision_cost (float, optional): traversal cost. Defaults to 1000..
      step_size (int, optional): step size (meter). Defaults to 0.25.
      stop_dist (float, optional): stop distance (meter). Defaults to 0.25.
      map_res (float, optional): map_res (meter/px). Defaults to 0.05.
    """
    self.step_size = step_size
    self.class_costs = class_costs
    self.collision_cost = collision_cost
    self.map_res = map_res
    self.stop_distance = stop_distance
    self.traversible = traversible

    self.du = int(self.step_size / self.map_res)
    self.fmm_dist = None
    self.fmm_cost = None

    self.large_number = 10000000


  def set_goal_map(
    self,
    goal_map: np.ndarray,
    collision_map: Optional[np.ndarray] = None,
    allow_collision: bool = False
  ):
    if allow_collision:
      # euclidean distance
      traversible = np.ones_like(self.traversible * 1)
    else:
      # geodesic distance
      traversible = np.ma.masked_values(self.traversible * 1, 0)
    traversible[goal_map == 1] = 0
    dd = skfmm.distance(traversible, dx=1)
    dd = np.ma.filled(dd, np.max(dd) + 1)
    self.fmm_dist = dd
    # calculate cost map
    speed = np.ones_like(self.traversible, dtype=np.float32)
    for idx, class_cost in enumerate(self.class_costs):
      speed[self.traversible == idx] = 1. / class_cost
    speed[self.traversible > idx] = 1. / class_cost
    if collision_map is not None:
      speed[collision_map == 1] = 1. / self.collision_cost
    dd = skfmm.travel_time(traversible, speed, dx=1)
    self.fmm_cost = dd

  def plan_by_cost(self, state, stop_by_distance=False):
    dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
    mask = get_mask(dx, dy, self.du)
    dist_mask = get_dist(dx, dy, self.du)

    state = [int(x) for x in state]

    dist = np.pad(self.fmm_dist, self.du,
      'constant', constant_values=self.large_number)

    subset_dist = dist[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]
    
    cost = np.pad(self.fmm_cost, self.du,
      'constant', constant_values=self.large_number)
    
    subset_cost = cost[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]

    # debug
    # fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(18, 4))
    # ax0.imshow(self.traversible, cmap='viridis')
    # ax1.imshow(np.clip(dist, 0, 500), cmap='viridis')
    # ax2.imshow(np.clip(cost, 0, 500), cmap='viridis')
    # ax3.imshow(subset_cost, cmap='viridis')
    # plt.tight_layout()
    # canvas = image_utils.plt2np(fig)
    # plt.close('all')
    # cv2.imshow('fmm', canvas[...,::-1])

    subset_cost *= mask
    subset_cost += (1 - mask) * self.large_number

    subset_dist *= mask
    subset_dist += (1 - mask) * self.large_number

    # use distance or cost to judge if the goal reached
    if stop_by_distance:
      distance = subset_dist[self.du, self.du]
    else:
      distance = subset_cost[self.du, self.du]
    stop = distance < self.stop_distance / self.map_res

    subset_cost -= subset_cost[self.du, self.du]
    (stg_x, stg_y) = np.unravel_index(np.argmin(subset_cost), subset_cost.shape)

    stg_x = (stg_x + state[0] - self.du) # subtract padded dist
    stg_y = (stg_y + state[1] - self.du)

    return stg_x, stg_y, distance, stop
