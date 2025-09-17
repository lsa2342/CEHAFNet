""" Transform BASNet"""
import random

import cv2
import noise
import torch
from skimage import io, transform, color
import numpy as np
import math

'''
图片预处理
'''


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """
        初始化颜色抖动类，用于随机调整图像的亮度、对比度、饱和度和色调。
        Args:
            brightness (float): 亮度调整幅度，范围 [0, 1]，0表示不调整。
            contrast (float): 对比度调整幅度，范围 [0, 1]，0表示不调整。
            saturation (float): 饱和度调整幅度，范围 [0, 1]，0表示不调整。
            hue (float): 色调调整幅度，范围 [0, 0.5]，0表示不调整。
                        色调范围是 [0, 180]，所以最大调整幅度是 0.5。
        """
        assert 0 <= brightness <= 1, "brightness must be in range [0, 1]"
        assert 0 <= contrast <= 1, "contrast must be in range [0, 1]"
        assert 0 <= saturation <= 1, "saturation must be in range [0, 1]"
        assert 0 <= hue <= 0.5, "hue must be in range [0, 0.5]"

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def adjust_brightness(self, image, factor):
        """调整亮度."""
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    def adjust_contrast(self, image, factor):
        """调整对比度."""
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

    def adjust_saturation(self, image, factor):
        """调整饱和度."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def adjust_hue(self, image, factor):
        """
        调整色调。
        Args:
            image (ndarray): RGB格式的输入图像。
            factor (float): 色调调整值，范围[-0.5, 0.5]。
        Returns:
            ndarray: 调整后的RGB图像。
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + factor * 180) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def __call__(self, sample):
        """
        对输入样本的图像进行颜色抖动增强。
        Args:
            sample (dict): 包含 'image' 和 'label' 的字典。
        Returns:
            dict: 包含增强后 'image' 和未改动的 'label' 的字典。
        """
        image, label = sample['image'], sample['label']

        # 随机调整亮度
        if self.brightness > 0:
            factor = 1 + np.random.uniform(-self.brightness, self.brightness)
            image = self.adjust_brightness(image, factor)

        # 随机调整对比度
        if self.contrast > 0:
            factor = 1 + np.random.uniform(-self.contrast, self.contrast)
            image = self.adjust_contrast(image, factor)

        # 随机调整饱和度
        if self.saturation > 0:
            factor = 1 + np.random.uniform(-self.saturation, self.saturation)
            image = self.adjust_saturation(image, factor)

        # 随机调整色调
        if self.hue > 0:
            factor = np.random.uniform(-self.hue, self.hue)
            image = self.adjust_hue(image, factor)

        return {'image': image, 'label': label}


class RandomShadow(object):
    def __init__(self, intensity_range=(0.2, 0.5), prob_polygon=0.6, prob_perlin=0.4, blur_size=51):
        """
        随机阴影增强类
        Args:
            intensity_range (tuple): 阴影强度范围
            prob_polygon (float): 多边形阴影概率
            prob_perlin (float): Perlin 噪声阴影概率
            blur_size (int): 阴影模糊核大小
        """
        self.intensity_range = intensity_range
        self.prob_polygon = prob_polygon
        self.prob_perlin = prob_perlin
        self.blur_size = blur_size

    def generate_polygon(self, h, w):
        """随机生成不规则多边形阴影区域"""
        num_vertices = random.randint(3, 8)
        contour = np.array([(random.randint(0, w), random.randint(0, h)) for _ in range(num_vertices)], dtype=np.int32)
        return contour

    def generate_perlin_noise(self, size, scale=10, octaves=3):
        """生成Perlin噪声阴影区域"""
        noise_map = np.zeros(size, dtype=np.float32)
        for y in range(size[0]):
            for x in range(size[1]):
                noise_map[y, x] = noise.pnoise2(x / scale, y / scale, octaves=octaves)
        noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
        return noise_map

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w, _ = image.shape
        shadow_mask = np.zeros((h, w), dtype=np.float32)

        # 随机选择阴影生成方式
        shadow_type = random.choices(['polygon', 'perlin'], weights=[self.prob_polygon, self.prob_perlin])[0]

        if shadow_type == 'polygon':
            num_shadows = random.randint(1, 3)
            for _ in range(num_shadows):
                polygon = self.generate_polygon(h, w)
                cv2.fillPoly(shadow_mask, [polygon], 1)

        elif shadow_type == 'perlin':
            shadow_mask = self.generate_perlin_noise((h, w))

        # 高斯模糊边缘
        shadow_mask = cv2.GaussianBlur(shadow_mask, (self.blur_size, self.blur_size), 0)

        # 随机强度扰动
        intensity = random.uniform(*self.intensity_range)
        shadow_mask = shadow_mask[..., np.newaxis]
        image = image * (1 - shadow_mask * intensity)
        image = np.clip(image, 0, 255).astype(np.uint8)

        return {'image': image, 'label': label}


class RandomRotate(object):
    def __init__(self, angle_range):
        """
        :param angle_range: 旋转角度范围，例如(-30, 30)，表示随机选择 -30 到 30 度之间的角度进行旋转。
        """
        assert isinstance(angle_range, tuple) and len(angle_range) == 2
        self.angle_range = angle_range

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 随机选择一个角度
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])

        # 获取图像的中心点
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 对图像进行旋转
        image_rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

        # 对标签进行旋转，确保标签是整数类型（如果标签是分割图，通常是整型）
        label_rotated = cv2.warpAffine(label, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        if len(label.shape) == 3:  # 如果标签是多通道的
            label_rotated = cv2.warpAffine(label, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        else:  # 如果标签是单通道的
            label_rotated = cv2.warpAffine(label, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

        return {'image': image_rotated, 'label': label_rotated}


class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0,
                               preserve_range=True)

        return {'image': img, 'label': lbl}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        img = transform.resize(image, (new_h, new_w), mode='constant')
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        return {'image': img, 'label': lbl}


class CenterCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # print("h: %d, w: %d, new_h: %d, new_w: %d"%(h, w, new_h, new_w))
        assert ((h >= new_h) and (w >= new_w))

        h_offset = int(math.floor((h - new_h) / 2))
        w_offset = int(math.floor((w - new_w) / 2))

        image = image[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
        label = label[h_offset: h_offset + new_h, w_offset: w_offset + new_w]

        return {'image': image, 'label': label}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)

        image = image / np.max(image)
        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'image': torch.from_numpy(tmpImg),
                'label': torch.from_numpy(tmpLbl)}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                    np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                    np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                    np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                    np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                    np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                    np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

        elif self.flag == 1:  #with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                    np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                    np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                    np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'image': torch.from_numpy(tmpImg),
                'label': torch.from_numpy(tmpLbl)}
