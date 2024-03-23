import os
import numpy as np
import pandas as pd
import cv2
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics
import seaborn as sns
from PIL import Image
import shutil
from functools import reduce
from tqdm.auto import tqdm
import time
import nibabel as nib
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import skimage
from .preprocessing import *
from .metrics import *
from .predictor import *
from .bbox import *
from .plotting import *
from glob import glob

# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# Working Directory

# data_dir = 'Data/'
np.random.seed(1337)

model = 'SAM-vit_h'
# model = 'SAM-vit_b'
# model = 'SAM-vit_l'
# model = 'MedSAM-vit_b'

if 'MedSAM' in model:
  enhance = False
else:
  enhance = True

predictor = Predictor(model=model, device='cuda')

def convert_bbox(bb_current,image_width,image_height):
  x_center, y_center = float(bb_current[0]), float(bb_current[1])
  box_width, box_height = float(bb_current[2]), float(bb_current[3])

  # Convert to top left and bottom right coordinates
  x0 = int((x_center - box_width / 2) * image_width)
  y0 = int((y_center - box_height / 2) * image_height)
  x1 = int((x_center + box_width / 2) * image_width)
  y1 = int((y_center + box_height / 2) * image_height)
  bbox = []
  bbox.append([x0, y0, x1, y1])
  return np.array(bbox)

# get all imaging data
fp_img = glob('SAM_Crowdsourcing/input/*.png')
fp_img.sort()

# get all bbox data
fp_out = glob('SAM_Crowdsourcing/output/*.txt')
fp_out.sort()

# As per Adway's labelling convention
label_map = { 0: 1, 1: 2, 2: 3, 5: 4, 7: 5}

os.makedirs(f'SAM_Crowdsourcing/{model}-masks/', exist_ok=True)

for iter1, img_name in enumerate(tqdm(fp_img)):
  img = cv2.imread(img_name)
  indx = img_name.rfind('/')
  img_name_sub = img_name[indx+1:]
  # print(img_name_sub)
  # print(img.shape)
  img = preprocess_image(np.array(img), edge_enhance=enhance)
  predictor.set_image(img)
  seg_mask = np.zeros((img.shape[0],img.shape[1]))
  box_name = fp_out[iter1]
  with open (box_name,"r") as fp:
    for line in fp:
      splitted = line.split()
      bb_current = splitted[1:]
      maskNum = label_map[int(splitted[0])]
      bbox = convert_bbox(bb_current,img.shape[0],img.shape[1])
      m = predictor.generate_mask_from_bbox(bbox)
      seg_mask[m] = maskNum
  cv2.imwrite(f'SAM_Crowdsourcing/{model}-masks/'+img_name_sub, seg_mask)



#bb_current1 = [0.375, 0.4677734375, 0.484375, 0.373046875]
#bb_current2 = [0.5078125, 0.5849609375, 0.08203125, 0.115234375]
#bb_current3 = [0.7060546875, 0.6513671875, 0.271484375, 0.255859375]
#bbox = convert_bbox(bb_current1, img.shape[0],img.shape[1])

#m = predictor.generate_mask_from_bbox(bbox)
#cv2.imwrite('testSeg.png',(np.float32(m)*255))
#print(np.max(np.float32(m)))
#cv2.imwrite('testSeg.png',m)

