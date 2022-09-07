# --- built in ---
# --- 3rd party ---
import numpy as np
# --- my module ---


def add_poses(pose: np.ndarray, delta: np.ndarray):
  """Accumulate pose changes

  Args:
    pose (np.ndarray): pose from
    delta (np.ndarray): local pose to accumulate
  """
  pose = np.asarray(pose)
  delta = np.asarray(delta)
  x1, y1, o1 = pose[...,0], pose[...,1], pose[...,2]
  dx, dy, do = delta[...,0], delta[...,1], delta[...,2]

  r = (dx**2.0 + dy**2.0)**0.5 # distance
  p = np.arctan2(dy, dx) + o1

  x2 = x1 + r * np.cos(p)
  y2 = y1 + r * np.sin(p)
  o2 = o1 + do
  o2 = np.arctan2(np.sin(o2), np.cos(o2)) # norm to [-pi/2, pi/2]
  return np.stack([x2, y2, o2], axis=-1)


def subtract_poses(pose2: np.ndarray, pose1: np.ndarray):
  """Calculate pose change from pose2 -> pose1

  Args:
    pose2 (np.ndarray): pose to
    pose1 (np.ndarray): pose from
  """
  pose1 = np.asarray(pose1)
  pose2 = np.asarray(pose2)
  x1, y1, o1 = pose1[...,0], pose1[...,1], pose1[...,2]
  x2, y2, o2 = pose2[...,0], pose2[...,1], pose2[...,2]

  r = ((x1-x2)**2 + (y1-y2)**2)**0.5
  p = np.arctan2(y2-y1, x2-x1) - o1
  dx = r * np.cos(p)
  dy = r * np.sin(p)
  do = o2 - o1
  do = np.arctan2(np.sin(do), np.cos(do)) # norm to [-pi/2, pi/2]
  return np.stack([dx, dy, do], axis=-1)
