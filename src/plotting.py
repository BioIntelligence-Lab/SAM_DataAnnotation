import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from .bbox import *

def find_and_visualize_bboxes(ax, img, mask):
  # Plot image
  ax.imshow(img, cmap='gray')
  ax.imshow(mask, alpha=0.5)
  bboxes = find_bboxes(mask)
  # Add bboxes as patches
  for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    xwidth = x2 - x1
    ywidth = y2 - y1
    p = patches.Rectangle(
      (x1, y1), 
      xwidth, 
      ywidth,
      fc = 'none', 
      ec = 'red'
    )
    ax.add_patch(p)
  ax.axis('off')

def overlay_masks(*masks, cmap=cm.viridis):
  masks = np.array(masks)
  targets = np.arange(masks.shape[0]) + 1
  overlay = np.sum(masks * targets[:,None,None], axis=0)
  color_overlay = cmap(rescale_intensities(overlay).astype(np.uint8))
  return (rescale_intensities(color_overlay)*overlay[:,:,None]).astype(np.uint8)

def visualize(ax, img, mask, points = None, bbox = None, bg_points = None):
  # Plot image
  ax.imshow(img, cmap='gray')
  ax.imshow(mask, alpha=0.6)
  if isinstance(points, np.ndarray):
    ax.plot(points[:,0], points[:,1], 'r.')
  if isinstance(bg_points, np.ndarray):
    ax.plot(bg_points[:,0], bg_points[:,1], 'rx')
  if isinstance(bbox, np.ndarray):
    # Add bboxes as patches
    for b in bbox:
      x1, y1, x2, y2 = b
      xwidth = x2 - x1
      ywidth = y2 - y1
      p = patches.Rectangle(
        (x1, y1), 
        xwidth, 
        ywidth,
        fc = 'none', 
        ec = 'red'
      )
      ax.add_patch(p)
  ax.axis('off')