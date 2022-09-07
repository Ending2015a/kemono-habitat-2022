# --- built in ---
import os
import json
# --- 3rd party ---
import habitat
import numpy as np
# --- my module ---

class InfoRecorder():
  """This class is used to track the evaluation results of each episode"""
  def __init__(self, log_path):
    self.log_path = log_path
    os.makedirs(self.log_path, exist_ok=True)
    self.info = {
      'distance_to_goal': [],
      'success': [],
      'spl': [],
      'softspl': []
    }
    self.infos = []
    self.avg_info = {}

  def add(self, info):
    self.infos.append(info)
    for key in self.info.keys():
      self.info[key].append(info.get(key, 0.0))
    self._update()

  def _update(self):
    log_path = os.path.join(self.log_path, 'episodic_metrics.json')
    with open(log_path, 'w') as f:
      json.dump(self.infos, f, ensure_ascii=False, indent=2)

    self.avg_info = {}
    for k, v in self.info.items():
      self.avg_info[k] = np.mean(v).item()
    log_path = os.path.join(self.log_path, 'metrics.json')
    with open(log_path, 'w') as f:
      json.dump(self.avg_info, f, ensure_ascii=False, indent=2)
