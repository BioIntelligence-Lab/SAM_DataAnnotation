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

def dice(
  y_pred,
  y_true
):
  if y_pred.dtype != bool:
    y_pred = y_pred.astype(bool)
  if y_true.dtype != bool:
    y_true = y_true.astype(bool)
  return np.sum(y_pred & y_true)*2.0 / (np.sum(y_pred) + np.sum(y_true))

def confidence_interval(
  x,
  axis = 0,
  p = 0.05
):
  mean = np.mean(x, axis=axis)
  std = np.std(x, axis=axis)
  n = x.shape[axis]
  z = stats.norm.ppf(1-p/2)
  err = z*std/np.sqrt(n)
  return z, err, (mean - err, mean + err)