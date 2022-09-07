# --- built in ---
import re
from typing import Dict, Optional, Union, TypeVar
# --- 3rd party ---
import numpy as np
import habitat
from habitat_sim.scene import (
  SemanticLevel,
  SemanticRegion,
  SemanticObject
)
# --- my module ---
from kemono.semantic import utils

SemanticAnnotations = TypeVar("SemanticAnnotations")

class SemanticMapping():
  def __init__(
    self,
    goal_mapping: Dict[int, str],
    default_category: str = 'unknown'
  ):
    """SemanticMapping provides an interface for mapping
    dataset's semantic meanings to MP3D/MPCat40 dataset's
    semantic meanings

    Args:
      dataset (habitat.Dataset, optional): task dataset for parsing
        task goals. Defaults to None.
      default_category (str, optional): default mpcat40 category for
        unlabeled/unknown objects in HM3D. Defaults to 'unknown'.
    """
    self._semantics = None
    self._object_id_to_object: Dict[int, SemanticObject] = {}
    self._object_id_to_mpcat40_category_id: Dict[int, int] = {}
    self._get_mpcat40_category_id_map = None
    # ---
    self.category_mapping = CategoryMapping(default_category=default_category)
    self.goal_mapping = GoalMapping(goal_mapping)

  @property
  def semantics(self) -> SemanticAnnotations:
    self.assert_semantics()
    return self._semantics

  @property
  def has_semantics(self) -> bool:
    return self._semantics is not None

  def assert_semantics(self):
    assert self.has_semantics, \
      "Call `self.parse_semantics(env.sim.semantic_annotations())` " \
      "to setup semantic annotations."

  @property
  def object_id_to_object(self) -> Dict[int, SemanticObject]:
    self.assert_semantics()
    return self._object_id_to_object

  @property
  def object_id_to_mpcat40_category_id(self) -> Dict[int, int]:
    self.assert_semantics()
    return self._object_id_to_mpcat40_category_id

  def _reset_object_data(self):
    """Reset object data, the object data is differed for each scene"""
    self._object_id_to_object = {}
    self._object_id_to_mpcat40_category_id = {}
    self._get_mpcat40_category_id_map = None

  def parse_semantics(
    self,
    semantic_annotations: SemanticAnnotations,
    reset: bool = True
  ):
    """Set & parse semantic annotations
    To use some functionality, you need to call this method
    to setup the semantic annotations

    Args:
      semantic_annotation (SemanticAnnotations): semantic annotations.
      reset (bool, optional): reset parsed objects data.
    """
    # setup semantic annotations
    self._semantics = semantic_annotations
    if reset:
      self._reset_object_data()
    # parse semantic annotations
    for level in self.semantics.levels:
      self.parse_level(level)
    for region in self.semantics.regions:
      self.parse_region(region)
    for object in self.semantics.objects:
      self.parse_object(object)
    # mapping scene objects to mpcat40 categories
    self.parse_object_categories()

  def parse_level(self, level: SemanticLevel):
    for region in level.regions:
      self.parse_region(region)

  def parse_region(self, region: SemanticRegion):
    for object in region.objects:
      self.parse_object(object)
  
  def parse_object(self, object: SemanticObject):
    obj_id = int(object.id.split('_')[-1])
    self.object_id_to_object[obj_id] = object

  def parse_object_categories(self):
    self.assert_semantics()
    for obj_id, object in self.object_id_to_object.items():
      category_name = object.category.name()
      # get mpcat40 category definitions
      mapping = self.category_mapping
      mpcat40cat = mapping.get_mpcat40cat_by_category_name(category_name)
      self._object_id_to_mpcat40_category_id[obj_id] = \
        mpcat40cat.mpcat40index
    # setup object id to mpcat40 category id mapping (vectorized)
    self._get_mpcat40_category_id_map = \
      np.vectorize(self._object_id_to_mpcat40_category_id.get)

  def print_semantic_meaning(self, top_n: int=3):
    """Print top N semantic meanings

    Args:
      top_n (int, optional): number of semantic meanings to print.
        Defaults to 3.
    """
    self.assert_semantics()
    def _print_scene(scene, top_n=10):
      count = 0
      for region in scene.regions:
        for obj in region.objects:
          self.print_object_info(obj)
          count += 1
          if count >= top_n:
            return None
    _print_scene(self.semantics, top_n=top_n)

  def get_mpcat40_category_id_map(
    self,
    object_id_map: np.ndarray
  ) -> np.ndarray:
    """_summary_

    Args:
      object_id_map (np.ndarray): expecting a 2D image (h, w),
        where each element is the object ID of the current dataset.
        Usually this is the output from the semantic sensors.

    Returns:
      np.ndarray: mpcat40 semantic segmentation image (h, w) (category ID)
    """
    self.assert_semantics()
    return self._get_mpcat40_category_id_map(object_id_map)

  def get_categorical_map(
    self,
    obj_id_map: np.ndarray,
    rgb: bool = False,
    bgr: bool = False
  ) -> np.ndarray:
    """Create mpcat40 categorical map
    colorized or non-colorized (category ID map).
    If either `rgb` or `bgr` is set, a colorized map is plotted.
    Otherwise a category ID map is plotted.

    Args:
      obj_id_map (np.ndarray): expected a 2D image (h, w),
        where each element is the object ID of the current dataset.
        Usually this is the output from the semantic sensors.
      rgb (bool, optional): plot in RGB channel order. Defaults
        to False.
      bgr (bool, optional): plot in BGR channel order. Defaults
        to False.

    Returns:
        np.ndarray: mpcat40 semantic segmentation image (h, w, 3).
    """
    self.assert_semantics()
    id_map = self.get_mpcat40_category_id_map(obj_id_map)
    if rgb or bgr:
      # priority: rgb > bgr
      bgr = not rgb
      mapping = self.category_mapping
      return mapping.get_colorized_mpcat40_category_map(id_map, bgr=bgr)
    return id_map

  def colorize_categorical_map(
    self,
    cat_id_map: np.ndarray,
    rgb: bool = False,
    bgr: bool = False
  ) -> np.ndarray:
    """Return a colorized map from mpcat40 category ID map
    the cat_id_map can be the ground truth id map generated from
    `self.get_categorical_map` with `rgb`, `bgr` set to False, or
    from the RedNet prediction (must be mpcat40 label).

    Args:
      cat_id_map (np.ndarray): mpcat40 categorical map
      rgb (bool, optional): plot in RGB channel order. Defaults
        to False.
      bgr (bool, optional): plot in BGR channel order. Defaults
        to False.

    Returns:
        np.ndarray: _description_
    """
    # priority: rgb > bgr
    bgr = not rgb
    mapping = self.category_mapping
    return mapping.get_colorized_mpcat40_category_map(cat_id_map, bgr=bgr)

  def get(self, object_id: int) -> Optional[SemanticObject]:
    """Get SemanticObject by object ID

    Args:
      object_id (int): object's ID in HM3D dataset

    Returns:
      SemanticObject: the corresponding object's semantic meanings
        if the ID does not exist, then return None.
    """
    self.assert_semantics()
    return self.object_id_to_object.get(object_id, None)

  def get_mpcat40cat(self, object_id: int) -> utils.MPCat40Category:
    """Get mpcat40 category definitions by object ID"""
    self.assert_semantics()
    mpcat40cat_id = self.object_id_to_mpcat40_category_id[object_id]
    return self.category_mapping.get_mpcat40cat(mpcat40cat_id)

  def print_object_info(
    self,
    object: Union[int, SemanticObject],
    verbose=False
  ):
    """Print object's semantic meanings in HM3D dataset
    set `verbose`=True for its corresponding mpcat40 semantic
    meanings.

    Args:
      object (Union[int, SemanticObject]): object id or object.
      verbose (bool, optional): whether to print mpcat40 semantic
        meanings. Defaults to False.
    """
    self.assert_semantics()
    # verbose: print mpcat40 info
    if not isinstance(object, SemanticObject):
      _object = self.get(object)
      assert _object is not None, f"object not found: {object}"
      object = _object
    print("==================")
    print(
      f"Object ID: {object.id}\n"
      f"  * category: {object.category.index()}/{object.category.name()}\n"
      f"  * center: {object.aabb.center}\n"
      f"  * dims: {object.aabb.sizes}"
    )
    if verbose:
      obj_id = int(object.id.split('_')[-1])
      mpcat40cat = self.get_mpcat40cat(obj_id)
      print(f"  * mpcat40 category: {mpcat40cat.mpcat40index}/{mpcat40cat.mpcat40}")
      print(f"  * color: {mpcat40cat.hex}")

  def get_goal_category_id(
    self, goal: int
  ) -> int:
    """Return mpcat40 category ID.
    Note that to use this API, you need to pass `goal_mapping`
    when constructing SemanticMapping.

    Args:
      goal (int): goal ID
    
    Returns:
      int: goal ID in mpcat40 labels
    """
    assert self.goal_mapping is not None
    category_name = self.goal_mapping.get_goal_name(goal)
    return self.category_mapping.get_mpcat40_id_by_category_name(category_name)

  def get_goal_category_name(
    self, goal: int
  ) -> str:
    """Return mpcat40 category name.
    Note that to use this API, you need to pass `dataset`
    when constructing SemanticMapping.

    Args:
      goal (int): goal ID

    Returns:
      int: goal name in mpcat40 labels
    """
    category_name = self.goal_mapping.get_goal_name(goal)
    return self.category_mapping.get_mpcat40_name_by_category_name(category_name)

class CategoryMapping():
  def __init__(
    self,
    default_category: str = 'unknown',
  ):
    """HM3DCategoryMapping helps mapping the HM3D dataset ID to
    mpcat40 category id, name, definitoins.

    Args:
      bgr (bool, optional): channel order of the image. BGR or RGB.
        Defaults to True.
      default_category (str, optional): default mpcat40 category for
        unlabeled/unknown objects in HM3D. Defaults to 'unknown'.
    """
    self.default_category = default_category
    # HM3D category name to MP3D raw category name
    self.manual_mapping = utils.hm3d_manual_map

  def get_mpcat40cat_by_category_name(
    self,
    category_name: str
  ):
    category_name = category_name.lower().strip()
    # replace multi spaces with single space
    category_name = re.sub(r'\s+', ' ', category_name)
    # map category name by user defined mappings
    if category_name in self.manual_mapping.keys():
      category_name = self.manual_mapping[category_name]
    # if the category name does not exists in the mp3d categories
    # set it to `unknown`.
    if category_name not in utils.mp3d_category_map:
      category_name = self.default_category
    # get mpcat40 category definitions
    return utils.mp3d_category_map[category_name]

  def get_mpcat40cat(
    self,
    mpcat40cat_id: int
  ):
    """Get mpcat40 category definitions by mpcat40 index"""
    return utils.mpcat40categories[mpcat40cat_id]

  def get_mpcat40_id_by_category_name(
    self,
    category_name: str
  ):
    mpcat40cat = self.get_mpcat40cat_by_category_name(category_name)
    return mpcat40cat.mpcat40index

  def get_mpcat40_name_by_category_name(
    self,
    category_name: str
  ):
    mpcat40cat = self.get_mpcat40cat_by_category_name(category_name)
    return mpcat40cat.mpcat40

  def get_colorized_mpcat40_category_map(
    self,
    category_id_map: np.ndarray,
    bgr: bool = False
  ) -> np.ndarray:
    """Get category map (RGB/BGR)"""
    cat_map = utils.mpcat40_color_map_rgb[category_id_map]
    if bgr:
      cat_map = cat_map[...,::-1]
    return cat_map

class GoalMapping():
  def __init__(
    self,
    goal_mapping: Dict[int, str],
  ):
    self.goal_mapping = goal_mapping
    self.name_to_id_mapping: Dict[str, int] = {}
    self.id_to_name_mapping: Dict[int, str] = {}
    # ---
    self.parse_dataset_goals()

  def parse_dataset_goals(self):
    mapping = self.goal_mapping
    self.id_to_name_mapping = {k:v for k, v in mapping.items()}
    self.name_to_id_mapping = {v:k for k, v in mapping.items()}

  def get_goal_name(self, id: int) -> str:
    id = np.asarray(id).item()
    return self.id_to_name_mapping.get(id, None)
  
  def get_goal_id(self, name: str) -> int:
    return self.name_to_id_mapping.get(name, None)
  

