import random

import numpy as np
import torch
from PIL import Image

#---------------------------------------------------------#
#   Convert image to RGB to prevent grayscale errors during prediction.
#   Only RGB images are supported; all other types are converted.
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   Resize input image with aspect ratio preserved
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   Get current learning rate from optimizer
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   Set random seed for reproducibility
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   Set per-worker random seed for DataLoader
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    import torch

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if backbone == "mobilenet":
        # Use torchvision's official pretrained MobileNetV2 weights
        import torchvision.models as models
        print("Downloading MobileNetV2 pretrained weights from torchvision...")
        pretrained = models.mobilenet_v2(pretrained=True)
        save_path = os.path.join(model_dir, "mobilenet_v2.pth.tar")
        torch.save(pretrained.state_dict(), save_path)
        print("Weights saved to %s" % save_path)
    elif backbone == "xception":
        # Xception has no official torchvision pretrained weights.
        # Please obtain the weights manually and place them in model_dir.
        target_path = os.path.join(model_dir, "xception_pytorch_imagenet.pth")
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                "Xception pretrained weights not found at %s. "
                "Xception weights are not available via torchvision. "
                "Please download them manually or train from scratch with pretrained=False." % target_path
            )
        print("Xception weights found at %s" % target_path)
    else:
        raise ValueError("Unsupported backbone: %s. Expected 'mobilenet' or 'xception'." % backbone)