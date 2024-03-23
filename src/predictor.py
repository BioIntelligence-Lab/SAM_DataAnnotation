from segment_anything import sam_model_registry, SamPredictor
from .preprocessing import *
from .bbox import *
import torch

def sample_slices(
  mask,
  num_slices = 10,
  seed = 1337
):
  np.random.seed(seed)
  # Filter out slices w/o segmentation
  slice_sum = mask.reshape(-1, mask.shape[-1]).sum(axis=0)
  idx = np.where(slice_sum > 0)[0]
  # Weigh slices by occurence of segmentation
  proba = slice_sum[idx]/slice_sum[idx].sum()
  # Randomly select slices
  if idx.shape[0] < num_slices:
    try:
      slices = np.random.choice(idx, idx.shape[0], p=proba, replace=False)
    except:
      slices = np.random.choice(idx, idx.shape[0], replace=False)
  else:
    try:
      slices = np.random.choice(idx, num_slices, p=proba, replace=False)
    except:
      slices = np.random.choice(idx, num_slices, replace=False)
  return slices

# TODO: Too many points affect performance, use Kmeans to cluster groups of annotations into fewer (for curatation at scale!)
def sample_points(
  mask,
  num_points = 10,
  seed = 1337
):
  np.random.seed(seed)
  # Determine points corresponding to mask
  points = np.array(np.where(mask > 0)).transpose()[:,[1,0]]
  # Randomly select points
  if points.shape[0] < num_points:
    points = points[np.random.randint(0, points.shape[0], points.shape[0])]
  else:
    points = points[np.random.randint(0, points.shape[0], num_points)]
  return points

class Predictor:
  def __init__(self, model='MedSAM', device='cuda'):
    if model == 'SAM-vit_h' or model == 'SAM':
      model = sam_model_registry['vit_h'](checkpoint='checkpoints/sam_vit_h_4b8939.pth')
    elif model == 'SAM-vit_b':
      model = sam_model_registry['vit_b'](checkpoint='checkpoints/sam_vit_b_01ec64.pth')
    elif model == 'SAM-vit_l':
      model = sam_model_registry['vit_l'](checkpoint='checkpoints/sam_vit_l_0b3195.pth')
    elif model == 'MedSAM-vit_b' or model == 'MedSAM':
      model = sam_model_registry['vit_b'](checkpoint='checkpoints/medsam_vit_b.pth')
    else:
      raise ValueError('Whoops!')
    model.to(device=device)
    self.predictor = SamPredictor(model)
  
  def set_image(self, img):
    self.predictor.set_image(img)

  def set_torch_image(self, img):
    self.predictor.set_torch_image(img)

  def predict(self, *args, **kwargs):
    return self.predictor.predict(*args, **kwargs)
  
  def predict_torch(self, *args, **kwargs):
    return self.predictor.predict_torch(*args, **kwargs)
  
  def get_image_embdding(self):
    return self.predictor.get_image_embedding()
  
  @property
  def device(self):
    return self.predictor.device
  
  def reset_image(self):
    self.predictor.reset_image()

  def generate_mask_from_points(
    self,
    points,
    targets
  ):
    masks, scores, _ = self.predictor.predict(
      points,
      targets,
      multimask_output = True
    )
    return masks[np.argmax(scores)]

  def generate_mask_from_bbox(
    self,
    bbox
  ):
    # Prepare bounding boxes
    batched_bbox = preprocess_bbox(bbox)
    # Determine number of batches by number of bboxes
    n_steps = batched_bbox.shape[0]
    masks = []
    for i in range(n_steps):
      box = batched_bbox[i]
      mask_i, scores_i, _ = self.predictor.predict(
        box = box,
        multimask_output = True
      )
      masks += [mask_i[np.argmax(scores_i)]]
    masks = np.sum(masks, axis=0)
    masks[masks > 1] = 1
    return masks.astype(bool)

  def generate_mask_from_points_bbox(
    self,
    points,
    targets,
    bbox
  ):
    # Prepare bounding boxes
    batched_bbox = preprocess_bbox(bbox)
    # Determine number of batches by number of bboxes
    n_steps = batched_bbox.shape[0]
    masks = []
    for i in range(n_steps):
      box = batched_bbox[i]
      points_in_box, targets_in_box = points_inside_bbox(points, targets, box)
      mask_i, scores_i, _ = self.predictor.predict(
        points_in_box,
        targets_in_box,
        box = box,
        multimask_output = True
      )
      masks += [mask_i[np.argmax(scores_i)]]
    masks = np.sum(masks, axis=0)
    masks[masks > 1] = 1
    return masks.astype(bool)