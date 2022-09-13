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

# === constants ===
LARGE_NUMBER = 10000000

def get_mask(r):
  d = int(r) * 2 + 1
  mask = np.zeros((d, d)) + 1e-8
  rr, cc = skimage.draw.circle_perimeter(r, r, r)
  mask[rr, cc] = 1
  mask[r, r] = 1
  return mask

class FMMPlanner():
  def __init__(
    self,
    travel_map: np.ndarray,
    class_costs: List[float] = [1., 100000., 1000.],
    step_size: float = 5,
    non_walkable: int = 1,
    debug: bool = False
  ):
    """Fast-marching short-term goal planner

    Args:
      travel_map (np.ndarray): traversible map, np.int64 or np.bool
      class_costs (List[float], optional): traversal cost of each class.
        in default it's [free, obstacle, collision, ...].
        Defaults to [1., 100000., 1000.].
      step_size (int, optional): step size (px). Defaults to 5.
    """
    self.step_size = step_size
    self.class_costs = class_costs
    self.travel_map = travel_map
    self.non_walkable = non_walkable
    self.debug = debug

    self.du = int(self.step_size)
    self.fmm_dist = None
    self.fmm_cost = None
    self.mask = get_mask(self.du)

    self.large_number = LARGE_NUMBER

  def set_goal_map(
    self,
    goal_map: np.ndarray,
    allow_collision: bool = False
  ):
    if allow_collision:
      # euclidean distance
      traversible = np.ones_like(self.travel_map, dtype=np.float32)
    else:
      # geodesic distance
      traversible = np.ones_like(self.travel_map, dtype=np.float32)
      # set non walkable to 0
      traversible[self.travel_map >= self.non_walkable] = 0
      traversible = np.ma.masked_values(traversible, 0)
    traversible[goal_map == 1] = 0
    dd = skfmm.distance(traversible, dx=1)
    dd = np.ma.filled(dd, np.max(dd) + 1)
    self.fmm_dist = dd
    # calculate cost map
    speed = np.ones_like(self.travel_map, dtype=np.float32)
    for idx, class_cost in enumerate(self.class_costs):
      speed[self.travel_map == idx] = 1. / class_cost
    speed[self.travel_map > idx] = 1. / class_cost
    dd = skfmm.travel_time(traversible, speed, dx=1)
    self.fmm_cost = dd

  def plan_by_cost(self, state, cost_as_distance=True):
    mask = self.mask.copy()

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
    if self.debug:
      fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(18, 4))
      ax0.imshow(self.travel_map, cmap='viridis')
      ax1.imshow(np.log(dist+1), cmap='viridis')
      ax2.imshow(np.log(cost+1), cmap='viridis')
      ax3.imshow(np.log(subset_cost+1), cmap='viridis')

    subset_cost *= mask
    subset_cost += (1 - mask) * self.large_number

    subset_dist *= mask
    subset_dist += (1 - mask) * self.large_number


    # use distance or cost to judge if the goal reached
    if cost_as_distance:
      distance = subset_cost[self.du, self.du]
    else:
      distance = subset_dist[self.du, self.du]

    subset_cost -= subset_cost[self.du, self.du]
    (stg_x, stg_y) = np.unravel_index(np.argmin(subset_cost), subset_cost.shape)

    # debug
    if self.debug:
      ax3.scatter(stg_y, stg_x, c='r', s=12)
      plt.tight_layout()
      canvas = image_utils.plt2np(fig)
      plt.close('all')
      cv2.imshow('fmm', canvas[...,::-1])

    stg_x = (stg_x + state[0] - self.du) # subtract padded dist
    stg_y = (stg_y + state[1] - self.du)

    return stg_x, stg_y, distance
