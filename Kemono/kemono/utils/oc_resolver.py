# --- built in ---
import os
import re
# --- 3rd party ---
from omegaconf import OmegaConf
# --- my module ---
from kemono.utils import image as image_utils

def oc_path(*args):
  # filter None
  args = list(filter(lambda a: a is not None, args))
  # convert to str and trim spaces
  args = [str(a).strip() for a in args]
  # filter empty string
  args = list(filter(lambda a: a, args))
  return os.path.join(*args)

OmegaConf.register_new_resolver("path", oc_path)
