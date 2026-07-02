import os
import random
import warnings

import numpy as np
import cv2
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

from src.config import IMAGE_SIZE_SIAMESE, RANDOM_SEED, DILATION_KERNEL_SIZE, SIAMESE_WEIGHTS_PATH, DEVICE
from src.utils import seed_everything

seed_everything(RANDOM_SEED)

# INFERENCE
val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
    ToTensorV2()])

# SIAMESE RESNET-34 U-NET MODEL
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)

        self.encoder_init = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.up4 = Up(64 + 64, 32)

        self.out_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward_encoder(self, x):
        x_half = self.encoder_init(x)
        x0 = self.maxpool(x_half)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x_half, x1, x2, x3, x4

    def forward(self, pre_img, post_img):
        pre_x_half, pre_x1, pre_x2, pre_x3, pre_x4 = self.forward_encoder(pre_img)
        post_x_half, post_x1, post_x2, post_x3, post_x4 = self.forward_encoder(post_img)

        diff_x_half = torch.abs(post_x_half - pre_x_half)
        diff_x1 = torch.abs(post_x1 - pre_x1)
        diff_x2 = torch.abs(post_x2 - pre_x2)
        diff_x3 = torch.abs(post_x3 - pre_x3)
        diff_x4 = torch.abs(post_x4 - pre_x4)

        x = self.up1(diff_x4, diff_x3)
        x = self.up2(x, diff_x2)
        x = self.up3(x, diff_x1)
        x = self.up4(x, diff_x_half)

        x = self.out_up(x)
        logits = self.out_conv(x)
        return logits


# IMAGE LOADING / PREPROCESSING
def load_pre_post_tensors(pre_image_path, post_image_path, transform=None, resize_shape=(512, 512)):
    with rasterio.open(pre_image_path) as src:
        pre_img = src.read([1, 2, 3])
    with rasterio.open(post_image_path) as src:
        post_img = src.read([1, 2, 3])

    pre_img = np.transpose(pre_img, (1, 2, 0))
    post_img = np.transpose(post_img, (1, 2, 0))

    if resize_shape:
        pre_img = cv2.resize(pre_img, resize_shape, interpolation=cv2.INTER_LINEAR)
        post_img = cv2.resize(post_img, resize_shape, interpolation=cv2.INTER_LINEAR)

    if transform:
        combined_img = np.concatenate([pre_img, post_img], axis=-1)
        augmented = transform(image=combined_img)
        augmented_img = augmented["image"]
        pre_tensor = augmented_img[:3, :, :]
        post_tensor = augmented_img[3:, :, :]
    else:
        pre_tensor = torch.tensor(pre_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        post_tensor = torch.tensor(post_img, dtype=torch.float32).permute(2, 0, 1) / 255.0

    return pre_tensor, post_tensor

# MODEL LOADING

def load_change_detection_model(weights_path, device):
    model = SiameseUNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

# SINGLE-PAIR INFERENCE
def generate_change_prediction(pre_image_path, post_image_path, model, device):
    pre_tensor, post_tensor = load_pre_post_tensors(
        pre_image_path,
        post_image_path,
        transform=val_transform,
        resize_shape=(IMAGE_SIZE_SIAMESE, IMAGE_SIZE_SIAMESE),
    )

    with torch.no_grad():
        pre_batch = pre_tensor.unsqueeze(0).to(device)
        post_batch = post_tensor.unsqueeze(0).to(device)

        outputs = model(pre_batch, post_batch)
        pred_mask = torch.sigmoid(outputs).squeeze().cpu().numpy()

        pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
        _, thresh_mask = cv2.threshold(pred_mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        pred_mask_1024 = cv2.resize(thresh_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    return pred_mask_1024


def save_change_prediction(pred_mask_1024, output_path):
    cv2.imwrite(output_path, pred_mask_1024)


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    WEIGHTS_PATH = SIAMESE_WEIGHTS_PATH
    PRE_IMAGE_PATH = ""
    POST_IMAGE_PATH = ""
    OUTPUT_MASK_PATH = "change_pred.png"

    if not WEIGHTS_PATH or not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")
    if not PRE_IMAGE_PATH or not os.path.exists(PRE_IMAGE_PATH):
        raise FileNotFoundError(f"Pre-disaster image not found: {PRE_IMAGE_PATH}")
    if not POST_IMAGE_PATH or not os.path.exists(POST_IMAGE_PATH):
        raise FileNotFoundError(f"Post-disaster image not found: {POST_IMAGE_PATH}")

    model = load_change_detection_model(WEIGHTS_PATH, DEVICE)
    pred_mask_1024 = generate_change_prediction(PRE_IMAGE_PATH, POST_IMAGE_PATH, model, DEVICE)
    save_change_prediction(pred_mask_1024, OUTPUT_MASK_PATH)
    print(f"Saved prediction mask to: {OUTPUT_MASK_PATH}")