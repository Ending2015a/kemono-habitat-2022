# --- bulit in ---
import os
from typing import Dict, List, Union, Tuple
import dataclasses
# --- 3rd party ---
import numpy as np
import pandas as pd
from torch import nn
from contextlib import contextmanager
# --- my module ---

__all__ = [
  'MPCat40Category',
  'HM3DCategory',
  'mpcat40categories',
  'hm3dcategories',
  'mp3d_category_map',
  'hm3d_manual_map',
  'mpcat40_color_map_rgb',
  'mpcat40_meaningful_ids',
  'mpcat40_trivial_ids',
  'evaluate'
]

# === Loading mp3d hm3d semantic labels ===
LIB_PATH = os.path.dirname(os.path.abspath(__file__))
# See https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
MPCAT40_PATH = os.path.join(LIB_PATH, 'mpcat40.tsv')
CATEGORY_MAPPING_PATH = os.path.join(LIB_PATH, 'category_mapping.tsv')
HM3D_MANUAL_MAPPING_PATH = os.path.join(LIB_PATH, 'manual_mapping.csv')
MPCAT40_DF = pd.read_csv(MPCAT40_PATH, sep='\t')
CATEGORY_MAPPING_DF = pd.read_csv(CATEGORY_MAPPING_PATH, sep='\t')
HM3D_MANUAL_MAPPING_DF = pd.read_csv(HM3D_MANUAL_MAPPING_PATH)

MPCAT40_TRIVIAL_LABELS = [
  'void',
  'misc',
  'unlabeled'
]

def dataclass_factory(df, name):
  _class = dataclasses.make_dataclass(
    name,
    [
      (column_name, column_dtype)
      for column_name, column_dtype in
      zip(df.dtypes.index, df.dtypes.values)
    ]
  )
  cat_insts = []
  for index, row in df.iterrows():
    cat_inst = _class(**{
      column: row[column]
      for column in df.columns
    })
    cat_insts.append(cat_inst)
  return _class, cat_insts

MPCat40Category, mpcat40categories = \
  dataclass_factory(MPCAT40_DF, 'MPCat40Category')
HM3DCategory, hm3dcategories = \
  dataclass_factory(CATEGORY_MAPPING_DF, 'HM3DCategory')

hex2rgb = lambda hex: [int(hex[i:i+2], 16) for i in (1, 3, 5)]
hex2bgr = lambda hex: hex2rgb(hex)[::-1]


def get_mp3d_category_map() -> Dict[str, MPCat40Category]:
  # mapping by either category name
  # default: category name
  mpcat40_name_to_category = {
    mpcat40cat.mpcat40: mpcat40cat
    for mpcat40cat in mpcat40categories
  }
  hm3d_name_to_mpcat40 = {}
  for hm3dcat in hm3dcategories:
    hm3d_name_to_mpcat40[hm3dcat.category] = \
      mpcat40_name_to_category[hm3dcat.mpcat40]
    hm3d_name_to_mpcat40[hm3dcat.raw_category] = \
      mpcat40_name_to_category[hm3dcat.mpcat40]
  for mpcat40cat in mpcat40categories:
    hm3d_name_to_mpcat40[mpcat40cat.mpcat40] = \
      mpcat40_name_to_category[mpcat40cat.mpcat40]
  return hm3d_name_to_mpcat40

mp3d_category_map = get_mp3d_category_map()

def get_mpcat40_color_map(bgr: bool=False) -> np.ndarray:
  colors = [
    hex2rgb(mpcat40cat.hex) if not bgr else hex2bgr(mpcat40cat.hex)
    for mpcat40cat in mpcat40categories
  ]
  colors = np.asarray(colors, dtype=np.uint8)
  return colors

mpcat40_color_map_rgb = get_mpcat40_color_map(bgr=False)

def get_hm3d_manual_mapping() -> Dict[str, str]:
  hm3d_map = HM3D_MANUAL_MAPPING_DF.to_dict()
  sources = hm3d_map['source'].values()
  targets = hm3d_map['target'].values()
  hm3d_map = {}
  for src, tar in zip(sources, targets):
    hm3d_map[src.strip()] = tar.strip()
  return hm3d_map

hm3d_manual_map = get_hm3d_manual_mapping()

def get_mpcat40_label_lists(
  trivial_lists: List[str] = MPCAT40_TRIVIAL_LABELS
):
  meaningful_ids = []
  trivial_ids = []
  for idx, cat in enumerate(mpcat40categories):
    if cat.mpcat40 in trivial_lists:
      trivial_ids.append(idx)
    else:
      meaningful_ids.append(idx)
  return meaningful_ids, trivial_ids

mpcat40_meaningful_ids, mpcat40_trivial_ids = get_mpcat40_label_lists()
# totally we have 40 classes: 39 meaningful classes + 1 trivial class

@contextmanager
def evaluate(model: nn.Module):
  """Temporary switch model to evaluation mode"""
  training = model.training
  try:
    model.eval()
    yield model
  finally:
    if training:
      model.train()