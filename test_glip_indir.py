import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import os

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)

def save_imshow(img, caption, pathname):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig(pathname)

config_file = "configs/pretrain/glip_Swin_L.yaml"
weight_file = "MODEL/glip_large_model.pth"

# Use this command to evaluate the GLPT-L model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
# config_file = "configs/pretrain/glip_Swin_L.yaml"
# weight_file = "MODEL/glip_large_model.pth"

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)

src_dir = '/mnt/share_disk/dataset/epic-kitchens/epic-kitchens-download-scripts/EPIC-KITCHENS/P08/rgb_frames/P08_11'
dst_dir = './figs/'
filenames = os.listdir(src_dir)

for filename in filenames:

    pil_image = Image.open(os.path.join(src_dir,filename)).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    caption = 'hand'
    result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
    pathname = os.path.join(dst_dir,"bbox_"+filename)
    save_imshow(result, caption, pathname)