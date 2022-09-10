# --- built in ---
import os
import math
from collections import deque
from typing import List, Optional
# --- 3rd party ---
import habitat
import numpy as np
import cv2
import torch
import skimage
import skimage.morphology
# --- my module ---
from kemono.utils import pose as pose_utils
from kemono.utils import image as image_utils
from kemono.agent.fmm_planner import FMMPlanner

class UnTrapHelper:
  def __init__(self):
    self.total_id = 0
    self.epi_id = 0

  def reset(self):
    self.total_id += 1
    self.epi_id = 0

  def act(self):
    self.epi_id += 1
    if self.epi_id == 1:
      if self.total_id % 2 == 0:
        return 2
      else:
        return 3
    else:
      if self.total_id % 2 == 0:
        return 3
      else:
        return 2

class AgentController():
  def __init__(
    self,
    conf
  ):
    self.conf = conf
    self.map_res = self.conf.map_res
    self.map_size = self.conf.map_size
    self.map_width = self.map_size
    self.map_height = self.map_size

    self.untrap = UnTrapHelper()

    self.agent = None
    self.env = None
    self.stg = None

    self.curr_pose = None
    self.prev_pose = None
    self.delta_pose = None

    self.plan_window = None
    self.collision_map = None
    self.visited_map = None
    self.prev_blocked = 0
    self.col_width = self.conf.collision_width
    self.col_length = self.conf.collision_length
    self.col_dist = self.conf.collision_dist
    self.goal_found = False
    self.last_action = None
    self.action_deque = deque(maxlen=6)

  def set_agent(self, agent):
    self.agent = agent
    self.env = agent.env

  def set_need_replan(self):
    assert self.agent is not None
    self.agent.set_need_replan()

  def update_state(self, observations):
    x = observations['gps'][0]
    y = -observations['gps'][1]
    o = observations['compass'][0]
    if o > np.pi:
      o -= 2 * np.pi
    curr_pose = np.array([x, y, o], dtype=np.float32)

    # get delta pose
    self.prev_pose = self.curr_pose
    if self.prev_pose is not None:
      delta_pose = pose_utils.subtract_poses(curr_pose, self.prev_pose)
      curr_pose = pose_utils.add_poses(self.prev_pose, delta_pose)
    else:
      delta_pose = np.zeros((3,))

    self.delta_pose = delta_pose
    self.curr_pose = curr_pose

  def reset(self):
    self.stg = None

    self.curr_pose = np.zeros((3,))
    self.prev_pose = np.zeros((3,))
    self.delta_pose = np.zeros((3,))

    self.plan_window = None
    self.collision_map = np.zeros((self.map_size, self.map_size))
    self.visited_map = np.zeros((self.map_size, self.map_size))
    self.prev_blocked = 0
    self.col_width = self.conf.collision_width
    self.col_length = self.conf.collision_length
    self.col_dist = self.conf.collision_dist
    self.goal_found = False
    self.last_action = None
    self.action_deque = deque(maxlen=6)

  def act(
    self,
    nav_map: np.ndarray,
    goal_map: np.ndarray,
    center_pos: np.ndarray,
    goal_found: bool = False,
    struct_map: Optional[np.ndarray] = None,
    plan_act: Optional[int] = None
  ):
    """_summary_

    Args:
      nav_map (np.ndarray): navigation map (h, w, c)
      center_pos (np.ndarray): center position of the local map [x, y]
      goal_found (bool, optional): _description_. Defaults to False.

    Returns:
      _type_: _description_
    """
    action = plan_act
    action = self._get_action(
      nav_map = nav_map,
      goal_map = goal_map,
      center_pos = center_pos,
      goal_found = goal_found,
      struct_map = struct_map,
      plan_act = plan_act
    )
    self.action_deque.append(action)
    self.last_action = action
    action = {'action': action}
    return action


  def _get_action(
    self,
    nav_map: np.ndarray,
    goal_map: np.ndarray,
    center_pos: np.ndarray,
    goal_found: bool,
    struct_map: Optional[int],
    plan_act: Optional[int]
  ) -> int:
    nav_map = np.rint(nav_map)

    plan_shape = nav_map.shape
    assert plan_shape[0] == plan_shape[1], "map must be a square"
    plan_size = plan_shape[0]
    gx1, gx2, gy1, gy2 = self.get_plan_map_bound(center_pos, plan_size)
    plan_window = [gx1, gx2, gy1, gy2]
    self.plan_window = plan_window
    half = self.map_size * self.map_res / 2
    # calculate agent's current location on planning map
    curr_loc = (self.curr_pose[:2] + half) / self.map_res
    curr_loc = [int(curr_loc[1] - gx1), int(curr_loc[0] - gy1)]
    curr_loc = np.clip(curr_loc, 0, plan_size-1)
    # calculate agent's previous location on planning map
    prev_loc = (self.prev_pose[:2] + half) / self.map_res
    prev_loc = [int(prev_loc[1] - gx1), int(prev_loc[0] - gy1)]
    prev_loc = np.clip(prev_loc, 0, plan_size-1)

    # update internal maps
    self._update_visited_map(plan_window, prev_loc, curr_loc)

    self._update_collision_map()

    stg, stop = self._get_stg(
      nav_map = nav_map,
      goal_map = goal_map,
      curr_loc = curr_loc,
      plan_window = plan_window,
      goal_found = goal_found,
      struct_map = struct_map
    )

    self.stg = stg

    if stop and goal_found:
      return 0 # stop
    else:
      (stg_x, stg_y) = stg
      goal_ang = np.arctan2(stg_x - curr_loc[0], stg_y - curr_loc[1])
      curr_ang = self.curr_pose[2]
      diff_ang = curr_ang - goal_ang
      rel_ang = np.arctan2(np.sin(diff_ang), np.cos(diff_ang))
      turn_ang = np.deg2rad(self.conf.turn_angle)
      turn_thres = self.conf.turn_thres
      if rel_ang > turn_ang * turn_thres:
        action = 3 # right
      elif rel_ang < -turn_ang * turn_thres:
        action = 2 # left
      else:
        action = 1 # forward
    
    if self.prev_blocked >= self.conf.block_thres:
      if self.last_action == 1:
        action = self.untrap.act()
      else:
        action = 1
    
    action_list = list(self.action_deque)
    if (action_list == [2,3,2,3,2,3] or
        action_list == [3,2,3,2,3,2]):
      # the agent is stucked
      action = 1 # forward agent

    if plan_act is not None and action > 0:
      # action from external planner
      action = plan_act
    return action

  def _get_stg(
    self,
    nav_map: np.ndarray,
    goal_map: np.ndarray,
    curr_loc: np.ndarray,
    plan_window: np.ndarray,
    goal_found: bool,
    struct_map: Optional[np.ndarray]
  ):
    [gx1, gx2, gy1, gy2] = plan_window

    def add_boundary(image, pad=1, value=1):
      pad = [[pad, pad]] * 2 + [[0, 0]] * (len(image.shape)-2)
      return np.pad(image, pad, mode='constant', constant_values=value)

    if len(nav_map) == 2:
      nav_map = np.expand_dims(nav_map, axis=-1) # (h, w, c)
    
    nav_map = nav_map.astype(bool)
    if self.conf.occ_brush_size > 0:
      for ch in range(nav_map.shape[-1]):
        nav_map[..., ch] = skimage.morphology.binary_dilation(
          nav_map[..., ch],
          skimage.morphology.disk(self.conf.occ_brush_size)
        )

    collision_map = self.collision_map[gx1:gx2, gy1:gy2].copy()
    visited_map = self.visited_map[gx1:gx2, gy1:gy2].copy()

    travel_map = np.zeros(nav_map.shape[:2], dtype=np.int32)

    idx = 0
    # [free, stairs]
    for ch in self.conf.walkable_channels:
      travel_map[nav_map[..., ch]] = idx
      idx += 1
    # collisions
    travel_map[collision_map == 1] = idx
    idx += 1
    for ch in range(nav_map.shape[-1]):
      if ch not in self.conf.walkable_channels:
        travel_map[nav_map[..., ch]] = idx
        idx += 1
    # visited
    travel_map[visited_map == 1] = 0

    # set agent's surroundings as free space
    s = self.conf.pad_surroundings
    travel_map[curr_loc[0]-s:curr_loc[0]+s+1,
      curr_loc[1]-s:curr_loc[1]+s+1] = 0


    pad = self.conf.pad_border if not goal_found else 0
    travel_map = add_boundary(travel_map, value=0, pad=pad)
    goal_map = add_boundary(goal_map, value=0, pad=pad)

    if goal_found and self.conf.use_iter_dilate:
      goal = goal_map.copy()
      brush = skimage.morphology.disk(self.conf.goal_iter_brush_size)
      assert struct_map is not None
      struct_map = add_boundary(struct_map, value=0, pad=pad)
      for i in range(self.conf.goal_iter_num):
        goal = skimage.morphology.binary_dilation(goal, brush)
        goal[struct_map == 1] = False
      goal = np.logical_or(goal, goal_map)
    else:
      brush = skimage.morphology.disk(self.conf.goal_brush_size)
      goal = skimage.morphology.binary_dilation(goal_map, brush)

    goal = goal.astype(np.float32)

    # fast marching planner, used for short-term goal planning
    planner = FMMPlanner(
      travel_map = travel_map,
      class_costs = self.conf.class_costs,
      step_size = self.conf.step_size,
      stop_distance = self.conf.stop_distance,
      map_res = self.map_res,
      non_walkable = self.conf.non_walkable
    )

    planner.set_goal_map(goal_map, allow_collision=True)

    # since we pad 1 to the plan map
    stg_x, stg_y, distance, stop = planner.plan_by_cost(curr_loc + pad)

    if distance > self.conf.replan_thres:
      if not goal_found:
        self.set_need_replan()
      else:
        # goal found but cannot reach
        brush = skimage.morphology.disk(2)
        goal = skimage.morphology.binary_dilation(goal, brush)
        goal = goal.astype(np.float32)
        planner.set_goal_map(goal, allow_collision=True)
        # at least one path will be found
        stg_x, stg_y, distance, stop = \
          planner.plan_by_cost(curr_loc + pad, stop_by_distance=True)
    
    # remove padding
    stg_x, stg_y = stg_x - pad, stg_y - pad
    return (stg_x, stg_y), stop

  def _update_visited_map(
    self,
    plan_window: np.ndarray,
    prev_loc: np.ndarray,
    curr_loc: np.ndarray
  ):
    gx1, gx2, gy1, gy2 = plan_window
    self.visited_map[gx1:gx2, gy1:gy2] = image_utils.draw_line(
      self.visited_map[gx1:gx2, gy1:gy2],
      prev_loc, curr_loc
    )

  def _update_collision_map(self):
    last_action = self.last_action
    if last_action == 1:
      half = self.map_size * self.map_res / 2
      # world coord
      x1, y1, t1 = self.prev_pose
      x2, y2, _ = self.curr_pose
      x1, y1 = x1 + half, y1 + half
      x2, y2 = x2 + half, y2 + half

      if abs(x1 - x2) + abs(y1 - y2) < self.conf.collision_thres:

        cx = (x1 + self.map_res * self.col_dist * np.cos(t1)) / self.map_res
        cy = (y1 + self.map_res * self.col_dist * np.sin(t1)) / self.map_res

        # draw collisions on collision map
        rr, cc = skimage.draw.ellipse(
          cy, cx, self.col_width, self.col_length, rotation=-t1
        )
        #rr, cc = skimage.draw.disk((cy, cx), self.col_width)
        rr = np.clip(rr, 0, self.map_size-1)
        cc = np.clip(cc, 0, self.map_size-1)
        self.collision_map[rr, cc] = 1

        self.prev_blocked += 1
        self.col_width = min(self.col_width + 0.5, self.conf.max_collision_width)
        self.col_length = min(self.col_length + 0.5, self.conf.max_collision_length)
        self.col_dist = min(self.col_dist + 0.5, self.conf.max_collision_dist)
      else:
        if self.prev_blocked >= self.conf.block_thres:
          self.untrap.reset()
        self.prev_blocked = 0
        # self.col_width = self.conf.collision_width
        # self.col_length = self.conf.collision_length
        # self.col_dist = self.conf.collision_dist

  def get_plan_map_bound(
    self,
    center_pos: np.ndarray,
    plan_size: np.ndarray
  ) -> List[int]:
    """Get plan map bound

    Args:
      center_pos (np.ndarray): center coord of the plan map
      plan_size (np.ndarray): [w, h]

    Returns:
      List[int]: left, right, top, down
    """
    half = self.map_size * self.map_res / 2
    center_loc = (center_pos[:2] + half) / self.map_res
    loc_r, loc_c = int(center_loc[1]), int(center_loc[0])
    local_w, local_h = plan_size, plan_size
    full_w, full_h = self.map_size, self.map_size

    gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
    gx2, gy2 = gx1 + local_w, gy1 + local_h
    if gx1 < 0:
      gx1, gx2 = 0, local_w
    if gx2 > full_w:
      gx1, gx2 = full_w - local_w, full_w
    if gy1 < 0:
      gy1, gy2 = 0, local_h
    if gy2 > full_h:
      gy1, gy2 = full_h - local_h, full_h
    return [gx1, gx2, gy1, gy2]

