# --- built in ---
# --- 3rd party ---
import torch
from torch import nn
import numpy as np
# --- my module ---

__all__ = [
  'classIoU',
  'mIoU'
]

@torch.no_grad()
def classIoU(
  seg_pred: torch.Tensor,
  seg_gt: torch.Tensor,
  class_idx: int,
  eps: float = 1e-8
) -> float:
  """Compute per class Intersection of Union (IoU)

  Args:
    seg_pred (torch.Tensor): predicted class indices, expecting a
      2/3/4D image. torch.int64
    seg_gt (torch.Tensor): ground truth class indices, expecting a
      2/3/4D image. torch.int64
    class_idx (int): index to calculate IoU.
    eps (float, optional): a small number. Defaults to 1e-8.

  Returns:
    float: nan if the ground truth does not contain `class_idx`,
      otherwise a floating number.
  """
  true_class = seg_pred == class_idx
  true_label = seg_gt == class_idx
  if true_label.long().sum().item() == 0:
    return np.nan
  else:
    intersect = torch.logical_and(true_class, true_label)
    union = torch.logical_or(true_class, true_label)
    intersect = intersect.sum().float().item()
    union = union.sum().float().item()
    iou = (intersect + eps) / (union + eps)
    return iou

@torch.no_grad()
def mIoU(
  seg_pred: torch.Tensor,
  seg_gt: torch.Tensor,
  num_classes: int = 40,
  eps: float = 1e-8
) -> float:
  """Calculate the mean of Intersection of Union across all classes

  Args:
    seg_pred (torch.Tensor): predicted class indices, expecting a
      2/3/4D image. torch.int64
    seg_gt (torch.Tensor): ground truth class indices, expecting a
      2/3/4D image. torch.int64
    num_classes (int, optional): number of classes. Defaults to 40.
    eps (float, optional): epsilon. Defaults to 1e-8.

  Returns:
    float: nan if the IoU of classes are all nan, otherwise a floating
      number.
  """  
  # flatten tensors
  iou_per_class = []
  for idx in range(num_classes):
    class_iou = classIoU(seg_pred, seg_gt, idx, eps=eps)
    iou_per_class.append(class_iou)
  return np.nanmean(iou_per_class)