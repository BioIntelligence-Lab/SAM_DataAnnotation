import os
import numpy as np
import pandas as pd
import cv2
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import metrics
import seaborn as sns
from PIL import Image
import shutil
from functools import reduce
from tqdm.auto import tqdm
import time
import nibabel as nib
from segment_anything import sam_model_registry, SamPredictor 
import skimage

def rescale_intensities(
  img, 
  old_range = None, 
  new_range = (0.0, 255.0)
):
  """
  Rescales pixel values/intensities from old range to a new range. 
  Arguments:
    img: A `np.ndarray` object
    old_range: An ordered pair (2-tuple) of min and max pixel values from previous distritbution
    new_range: An ordered pair (2-tuple) of min and max pixel values from new distritbution
  Returns:
    An image as `np.ndarray` object with pixel values rescaled from old to new distribution
  """
  if old_range:
    old_min, old_max = old_range
  else:
    old_min, old_max = img.min(), img.max()
  new_min, new_max = new_range
  return ((img - old_min)/(old_max - old_min))*(new_max - new_min) + new_min

def preprocess_image(
  img,
  edge_enhance = True
):
  if img.ndim == 3 and img.shape[-1] != 3:
    raise ValueError(f'Image must either be grayscale or RGB. Input image shape: {img.shape}')
  # Convert grayscale image to RGB
  if img.ndim == 2:
    img = np.expand_dims(img, axis=-1)
    img = np.concatenate((img,)*3, axis=-1)
  # Resample image to [0,255]
  img = rescale_intensities(img.astype(np.float32))
  # Edge enhance
  if edge_enhance:
    img += rescale_intensities(skimage.filters.sobel(img))
    return rescale_intensities(img).astype(np.uint8)
  else:
    return img.astype(np.uint8)

def preprocess_mask(
  mask,
  target_class = 1,
  strict = False
):
  if mask.dtype == bool:
    mask = mask.astype(np.float64)
  # Filter out all other classes within segmentation mask
  if strict:
    mask[mask != target_class] = 0
    mask[mask == target_class] = 1
  else:
    mask[mask < target_class] = 0
    mask[mask >= target_class] = 1
  return mask

def preprocess_bbox(
  bbox
):
  if bbox.ndim == 1:
    bbox = np.expand_dims(bbox, axis=0)
  if bbox.ndim > 2 or bbox.shape[-1] != 4:
    raise ValueError('Invalid shape! Bounding box input must be in XYXY format with shape: (4,) or (N,4)')
  return bbox

def invert_mask(
  mask
):
  if mask.dtype == bool:
    return np.invert(mask)
  else:
    return np.abs(mask - 1)
