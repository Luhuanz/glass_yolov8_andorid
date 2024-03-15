import cv2
from albumentations import (
    Compose, OneOf, GaussianBlur, GaussNoise, RandomBrightnessContrast,
    HueSaturationValue, ISONoise, Sharpen, RandomGamma
)
import os
import shutil
from glob import glob

# 定义增强操作
augmentations = Compose([
    OneOf([
        GaussNoise(var_limit=(20.0, 80.0), p=1),
        ISONoise(intensity=(0.1, 0.3), color_shift=(0.01, 0.05), p=1),
        Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1)
    ], p=1),
    GaussianBlur(blur_limit=(5, 11), p=1),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
])

def augment_image(image_path, label_dir, num_augmented=4):
    image = cv2.imread(image_path)
    # 检查图片是否成功读取
    if image is None:
        print(f"Warning: Unable to read image {image_path}. Skipping...")
        return  # 跳过当前图片的处理

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    base_image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.dirname(image_path)
    label_path = os.path.join(label_dir, base_image_name + '.txt')

    for i in range(num_augmented):
        augmented = augmentations(image=image)
        image_augmented = augmented['image']
        augmented_image_path = os.path.join(image_dir, f"{base_image_name}_aug_{i}.jpg")
        image_augmented_bgr = cv2.cvtColor(image_augmented, cv2.COLOR_RGB2BGR)
        cv2.imwrite(augmented_image_path, image_augmented_bgr)
        augmented_label_path = os.path.join(label_dir, f"{base_image_name}_aug_{i}.txt")
        shutil.copy(label_path, augmented_label_path)

image_dir = r'C:\Users\Admin\Desktop\mydataset-seg\mydataset-seg\images\train'
label_dir = r'C:\Users\Admin\Desktop\mydataset-seg\mydataset-seg\labels\train'
image_paths = glob(os.path.join(image_dir, '*.jpg'))

for image_path in image_paths:
    augment_image(image_path, label_dir)
