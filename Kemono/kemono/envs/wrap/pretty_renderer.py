# --- built in ---
from typing import (
  Any,
  Optional,
  Union,
  List,
  Dict
)
# --- 3rd party ---
import habitat
import numpy as np
import gym
import cv2
# --- my module ---

class PrettyRenderer(gym.Wrapper):
  objectgoal_key = 'objectgoal'
  def __init__(
    self,
    env: habitat.RLEnv,
    canvas_config: List[List[Dict[str, Any]]],
    goal_mapping: Dict[int, str],
    backend: str = 'cv2',
    dpi: int = 200
  ):
    super().__init__(env=env)
    self._cached_obs = None
    self._cached_canvas = None
    self.canvas_config = canvas_config
    self.goal_mapping = goal_mapping
    self.backend = backend
    self.dpi = dpi

  def step(self, *args, **kwargs):
    self._cached_canvas = None
    obs, rew, done, info = self.env.step(*args, **kwargs)
    self._cached_obs = obs
    return obs, rew, done, info

  def reset(self, *args, **kwargs):
    self._cached_canvas = None
    obs = self.env.reset(*args, **kwargs)
    self._cached_obs = obs
    return obs

  def render(self, mode='human', scale=1.0):
    if mode == 'interact':
      return self.env.render(mode=mode)
    if self._cached_canvas is None:
      canvas = self.render_canvas()
      self._cached_canvas = canvas
    canvas = self._cached_canvas
    if scale != 1.0:
      h, w, _ = canvas.shape
      canvas = cv2.resize(canvas, (int(w*scale), int(h*scale)))
    if mode == 'human':
      cv2.imshow('Habitat', canvas[...,::-1])
      cv2.waitKey(1)
    return canvas

  def render_canvas(self):
    
    canvas = []
    for idx, configs in enumerate(self.canvas_config):
      first_row = (idx == 0)
      if self.backend == 'cv2':
        canvas.append(self.render_row_cv2(configs, first_row))
      else:
        canvas.append(self.render_row_matplotlib(configs, first_row))
    # concat all canvas (verticle)
    max_width = 0
    for c in canvas:
      max_width = max(max_width, c.shape[1])
    
    for idx in range(len(canvas)):
      width = canvas[idx].shape[1]
      pad = max_width - width
      if pad == 0:
        continue
      l_pad = pad//2
      r_pad = pad - l_pad
      canvas[idx] = (
        np.pad(c, ((0, 0), (l_pad, r_pad), (0, 0)), constant_values=255)
      )
    canvas = np.concatenate(canvas, axis=0)
    return canvas

  def render_row_cv2(self, configs, first_row=True):
    goal_id = self._cached_obs[self.objectgoal_key]
    goal_name = self.goal_mapping[np.asarray(goal_id).item()]
    canvas = []
    for idx, config in enumerate(configs):
      key = config['input']
      size = config['size']
      label = config['label']
      image = self._cached_obs.get(key, None)
      if image is None:
        print(f'PrettyRender: Key \'{key}\' not found.')
        continue
      image = cv2.resize(image, size, cv2.INTER_NEAREST)
      image = np.pad(image, ((2, 2), (2, 2), (0, 0)), constant_values=100)
      image = np.pad(image, ((100, 20), (20, 20), (0,0)), constant_values=255)
      if idx == 0 and first_row:
        label = label + f' (Goal: {goal_name})'
      labelsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, 1.3, 1)[0]
      woff = (image.shape[1] - labelsize[0]) // 2
      hoff = (100 + labelsize[1]) // 2
      image = cv2.putText(image, label, (woff, hoff), cv2.FONT_HERSHEY_TRIPLEX,
        1.3, (0, 0, 0), 1, cv2.LINE_AA)
      canvas.append(image)
    canvas = np.concatenate(canvas, axis=1)
    return canvas

  def render_row_matplotlib(self, configs, first_row=True):
    # trying to use matplotlib as the backend
    # rendering mode is human or rgb_array
    goal_id = self._cached_obs[self.objectgoal_key]
    goal_name = self.goal_mapping[np.asarray(goal_id).item()]
    import matplotlib.pyplot as plt
    import io

    num_plots = len(configs)

    # pre-calculate aspect ratio
    ws, hs = [], []
    for config in configs:
      size = config['size']
      ws.append(size[0])
      hs.append(size[1])
    ws = np.asarray(ws)
    hs = np.asarray(hs)
    width_ratios = ws / ws.min()
    w = ws.sum()
    h = hs.max()
    if first_row:
      h += 150 # suptitle
    figsize = plt.figaspect(h/w)

    fig, axs = plt.subplots(
      figsize=figsize,
      ncols=num_plots,
      nrows=1,
      dpi=self.dpi,
      gridspec_kw={'width_ratios': width_ratios}
    )

    # set figure title
    if first_row:
      fig.suptitle(f'Goal: {goal_name}', fontsize='xx-large')
    for idx, (config, ax) in enumerate(zip(configs, axs)):
      label = config['label']
      size = tuple(config['size'])
      key = config['input']
      image = self._cached_obs.get(key, None)
      if image is None:
        print(f'PrettyRender: Key \'{key}\' not found.')
        continue
      image = cv2.resize(image, size, cv2.INTER_NEAREST)
      # plot image
      ax.imshow(image, interpolation='nearest')
      ax.set_title(label, fontsize='x-large')
      ax.grid(False)
      # disable ticks and preserve border line
      ax.set_xticks([])
      ax.set_yticks([])

    plt.tight_layout()
    # convert matplotlib plot to np.ndarray
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    canvas = np.reshape(
      np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
      (int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
    )[...,:3] # rgba -> rgb
    io_buf.close()
    plt.close('all')
    return canvas
