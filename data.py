import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance, ImageFilter
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates


# ============================================================================
# Advanced Augmentation Classes
# ============================================================================

class RandomColorJitter:
    """Aggressive color jittering for OOD robustness"""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask

        if random.random() > 0.5:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if random.random() > 0.5:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)

        if random.random() > 0.5:
            factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            image = ImageEnhance.Color(image).enhance(factor)

        if random.random() > 0.5:
            image_np = np.array(image)
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] += random.uniform(-self.hue * 180, self.hue * 180)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
            image_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            image = Image.fromarray(image_np)

        return image, mask


class RandomGammaCorrection:
    """Gamma correction for different lighting"""

    def __init__(self, gamma_range=(0.5, 2.0), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            image_np = np.array(image).astype(np.float32) / 255.0
            image_np = np.power(image_np, gamma)
            image = Image.fromarray((image_np * 255).astype(np.uint8))
        return image, mask


class RandomColorCast:
    """Random color cast for white balance variations"""

    def __init__(self, intensity=0.25, p=0.4):
        self.intensity = intensity
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image_np = np.array(image).astype(np.float32)
            cast = np.random.uniform(1 - self.intensity, 1 + self.intensity, 3)
            image_np *= cast
            image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
        return image, mask


class ElasticTransform:
    """Elastic deformation for tissue variations"""

    def __init__(self, alpha=30, sigma=5, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image_np = np.array(image)
            mask_np = np.array(mask)

            shape = image_np.shape[:2]
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

            distorted_image = np.zeros_like(image_np)
            for i in range(3):
                distorted_image[:, :, i] = map_coordinates(
                    image_np[:, :, i], indices, order=1, mode='reflect'
                ).reshape(shape)

            distorted_mask = map_coordinates(
                mask_np, indices, order=0, mode='reflect'
            ).reshape(shape)

            image = Image.fromarray(distorted_image.astype(np.uint8))
            mask = Image.fromarray(distorted_mask.astype(np.uint8))

        return image, mask


class RandomSpecularReflection:
    """Add specular reflections (bright spots) for endoscopy"""

    def __init__(self, num_spots_range=(1, 3), p=0.3):
        self.num_spots_range = num_spots_range
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image_np = np.array(image).astype(np.float32)
            h, w = image_np.shape[:2]

            num_spots = random.randint(*self.num_spots_range)
            for _ in range(num_spots):
                cx, cy = random.randint(0, w-1), random.randint(0, h-1)
                radius = random.randint(20, 60)

                y, x = np.ogrid[:h, :w]
                spot_mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius ** 2))

                for i in range(3):
                    image_np[:, :, i] = image_np[:, :, i] * (1 - spot_mask * 0.8) + \
                                        255 * spot_mask * 0.8

            image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))

        return image, mask


class RandomShadow:
    """Add random shadows for lighting variations"""

    def __init__(self, intensity_range=(0.3, 0.7), p=0.4):
        self.intensity_range = intensity_range
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image_np = np.array(image).astype(np.float32)
            h, w = image_np.shape[:2]

            shadow_mask = np.ones((h, w), dtype=np.float32)
            num_points = random.randint(3, 6)
            points = [(random.randint(0, w-1), random.randint(0, h-1)) for _ in range(num_points)]

            shadow_intensity = random.uniform(*self.intensity_range)
            cv2.fillPoly(shadow_mask, [np.array(points)], shadow_intensity)
            shadow_mask = gaussian_filter(shadow_mask, sigma=50)

            for i in range(3):
                image_np[:, :, i] *= shadow_mask

            image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))

        return image, mask


# ============================================================================
# Basic Geometric Augmentations
# ============================================================================

def cv_random_flip(img, label):
    """Random horizontal flip"""
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomRotation(image, label, degrees=30):
    """Random rotation"""
    if random.random() > 0.7:
        angle = random.uniform(-degrees, degrees)
        image = image.rotate(angle, Image.BICUBIC)
        label = label.rotate(angle, Image.NEAREST)
    return image, label


def randomCrop(image, label, crop_ratio=0.85):
    """Random crop with resize"""
    if random.random() > 0.5:
        w, h = image.size
        crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)

        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)

        image = image.crop((left, top, left + crop_w, top + crop_h))
        label = label.crop((left, top, left + crop_w, top + crop_h))

        image = image.resize((w, h), Image.BICUBIC)
        label = label.resize((w, h), Image.NEAREST)

    return image, label


def randomScale(image, label, scale_range=(0.75, 1.25)):
    """Random scaling"""
    if random.random() > 0.5:
        scale = random.uniform(*scale_range)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)

        image = image.resize((new_w, new_h), Image.BICUBIC)
        label = label.resize((new_w, new_h), Image.NEAREST)

        if scale > 1.0:
            left, top = (new_w - w) // 2, (new_h - h) // 2
            image = image.crop((left, top, left + w, top + h))
            label = label.crop((left, top, left + w, top + h))
        else:
            from torchvision.transforms import functional as TF
            pad_w, pad_h = (w - new_w) // 2, (h - new_h) // 2
            image = TF.pad(image, (pad_w, pad_h, w - new_w - pad_w, h - new_h - pad_h))
            label = TF.pad(label, (pad_w, pad_h, w - new_w - pad_w, h - new_h - pad_h))

    return image, label


def randomGaussianNoise(image, std_range=(5, 25)):
    """Add Gaussian noise"""
    if random.random() > 0.5:
        image_np = np.array(image).astype(np.float32)
        std = random.uniform(*std_range)
        noise = np.random.normal(0, std, image_np.shape)
        image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_np)
    return image


def randomBlur(image):
    """Random blur"""
    if random.random() > 0.5:
        blur_type = random.choice(['gaussian', 'motion'])
        if blur_type == 'gaussian':
            radius = random.choice([1, 2, 3])
            image = image.filter(ImageFilter.GaussianBlur(radius))
        else:
            size = random.choice([3, 5, 7])
            kernel = np.zeros((size, size))
            kernel[int((size - 1) / 2), :] = np.ones(size) / size
            image_np = cv2.filter2D(np.array(image), -1, kernel)
            image = Image.fromarray(image_np)
    return image


# ============================================================================
# Dataset Class with Advanced Augmentation
# ============================================================================

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, mode='train', augmentation_level='light'):
        """
        Args:
            image_root: Path to images
            gt_root: Path to masks
            trainsize: Target image size
            mode: 'train' or 'val'
            augmentation_level: 'aggressive', 'medium', or 'light'
        """
        self.trainsize = trainsize
        self.mode = mode
        self.augmentation_level = augmentation_level

        # Load file paths
        self.images = [image_root + f for f in os.listdir(image_root)
                       if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') ]
        self.gts = [gt_root + f for f in os.listdir(gt_root)
                    if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.filter_files()
        self.size = len(self.images)

        print(f"[{mode.upper()}] Loaded {self.size} images with '{augmentation_level}' augmentation")

        # Initialize advanced augmentations ONLY for training
        if mode == 'train':
            if augmentation_level == 'aggressive':
                self.color_aug = RandomColorJitter(0.4, 0.4, 0.4, 0.15, p=0.8)
                self.gamma_aug = RandomGammaCorrection((0.5, 2.0), p=0.5)
                self.colorcast_aug = RandomColorCast(0.25, p=0.4)
                self.elastic_aug = ElasticTransform(30, 5, p=0.3)
                self.specular_aug = RandomSpecularReflection((1, 3), p=0.3)
                self.shadow_aug = RandomShadow((0.3, 0.7), p=0.4)

            elif augmentation_level == 'medium':
                self.color_aug = RandomColorJitter(0.3, 0.3, 0.3, 0.1, p=0.7)
                self.gamma_aug = RandomGammaCorrection((0.7, 1.5), p=0.4)
                self.colorcast_aug = RandomColorCast(0.15, p=0.3)
                self.elastic_aug = None
                self.specular_aug = RandomSpecularReflection((1, 2), p=0.2)
                self.shadow_aug = RandomShadow((0.4, 0.7), p=0.3)

            else:  # light
                self.color_aug = RandomColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5)
                self.gamma_aug = None
                self.colorcast_aug = None
                self.elastic_aug = None
                self.specular_aug = None
                self.shadow_aug = None
        else:
            # Validation mode: no augmentation
            self.color_aug = None
            self.gamma_aug = None
            self.colorcast_aug = None
            self.elastic_aug = None
            self.specular_aug = None
            self.shadow_aug = None

        # Standard transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # Load image and mask
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        if self.mode == 'train':
            # Apply augmentations in order

            # 1. Advanced color/intensity augmentations (CRITICAL for OOD!)
            if self.color_aug:
                image, gt = self.color_aug(image, gt)
            if self.gamma_aug:
                image, gt = self.gamma_aug(image, gt)
            if self.colorcast_aug:
                image, gt = self.colorcast_aug(image, gt)

            # 2. Geometric augmentations
            image, gt = cv_random_flip(image, gt)
            image, gt = randomRotation(image, gt, degrees=30)
            image, gt = randomCrop(image, gt, crop_ratio=0.85)
            image, gt = randomScale(image, gt, scale_range=(0.75, 1.25))

            # 3. Elastic deformation
            if self.elastic_aug:
                image, gt = self.elastic_aug(image, gt)

            # 4. Medical-specific augmentations
            if self.specular_aug:
                image, gt = self.specular_aug(image, gt)
            if self.shadow_aug:
                image, gt = self.shadow_aug(image, gt)

            # 5. Noise/blur (apply to image only)
            image = randomGaussianNoise(image, std_range=(5, 25))
            image = randomBlur(image)

        # Convert to tensors
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt

    def filter_files(self):
        """Filter images and GTs to ensure matching sizes"""
        assert len(self.images) == len(self.gts), "Mismatch between images and masks"
        images, gts = [], []

        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)

        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize,
               mode='train', augmentation_level='light',
               shuffle=True, num_workers=4, pin_memory=True):
    """
    Create data loader with advanced augmentation.

    Args:
        image_root: Path to images
        gt_root: Path to masks
        batchsize: Batch size
        trainsize: Target size
        mode: 'train' or 'val'
        augmentation_level: 'aggressive' (< 500 images), 'medium' (500-1000), 'light' (> 1000)
        shuffle: Shuffle data
        num_workers: Number of workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader
    """
    dataset = SalObjDataset(image_root, gt_root, trainsize, mode, augmentation_level)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle if mode == 'train' else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if mode == 'train' else False
    )

    return data_loader


# ============================================================================
# Test Dataset (No Augmentation)
# ============================================================================

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root)
                       if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root)
                    if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index = (self.index + 1) % self.size
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


if __name__ == '__main__':
    # Test the data loader
    print("Testing Advanced Data Augmentation Pipeline")
    print("=" * 60)

    # Simulated paths
    image_root = './dataset/train/Kvasir-SEG/images/'
    gt_root = './dataset/train/Kvasir-SEG/masks/'

    if os.path.exists(image_root):
        # Test different augmentation levels
        for level in ['aggressive', 'medium', 'light']:
            print(f"\nTesting '{level}' augmentation:")
            loader = get_loader(image_root, gt_root,
                                batchsize=4, trainsize=512,
                                mode='train', augmentation_level=level)

            images, masks = next(iter(loader))
            print(f"  Batch shape: {images.shape}")
            print(f"  Masks shape: {masks.shape}")
    else:
        print("Dataset path not found. Please update paths.")