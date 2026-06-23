#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom transforms for PaddleOCR data augmentation.
Includes MotionBlur and GaussianNoise transforms.
"""

import cv2
import numpy as np
import random

class MotionBlur(object):
    """
    Apply motion blur to the image to simulate video/camera motion blur.
    """
    def __init__(self, kernel_size=[3, 7], p=0.3, **kwargs):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            image = data['image']
            # Chọn kích thước kernel ngẫu nhiên và đảm bảo là số lẻ
            min_k, max_k = self.kernel_size[0], self.kernel_size[1]
            odd_kernels = [k for k in range(min_k, max_k + 1) if k % 2 == 1]
            if not odd_kernels:
                return data
            ksize = random.choice(odd_kernels)
            
            # Tạo motion blur kernel nằm ngang
            kernel = np.zeros((ksize, ksize))
            kernel[int((ksize - 1) / 2), :] = 1
            
            # Xoay kernel theo một góc ngẫu nhiên để tạo các góc mờ khác nhau
            angle = random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((ksize / 2.0, ksize / 2.0), angle, 1.0)
            kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
            
            # Normalize kernel
            kernel_sum = np.sum(kernel)
            if kernel_sum > 0:
                kernel = kernel / kernel_sum
            else:
                kernel = np.ones((ksize, ksize)) / (ksize * ksize)
                
            data['image'] = cv2.filter2D(image, -1, kernel)
        return data


class GaussianNoise(object):
    """
    Add Gaussian noise to the image.
    """
    def __init__(self, var_limit=[10, 50], p=0.2, **kwargs):
        self.var_limit = var_limit
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            image = data['image']
            # Tính toán độ lệch chuẩn từ phương sai ngẫu nhiên
            var = random.uniform(self.var_limit[0], self.var_limit[1])
            sigma = var ** 0.5
            
            # Sinh nhiễu Gaussian
            noise = np.random.normal(0, sigma, image.shape)
            
            # Cộng nhiễu và clip giá trị trong khoảng [0, 255]
            noisy_image = image.astype(np.float32) + noise
            data['image'] = np.clip(noisy_image, 0, 255).astype(image.dtype)
        return data


def register_custom_transforms():
    """
    Đăng ký custom transforms vào module ppocr.data.imaug của PaddleOCR.
    Điều này cho phép PaddleOCR tự động khởi tạo các transform này từ file config YAML.
    """
    try:
        import sys
        # Đảm bảo ppocr đã được import trước khi đăng ký
        if 'ppocr.data.imaug' in sys.modules:
            imaug_module = sys.modules['ppocr.data.imaug']
            setattr(imaug_module, 'MotionBlur', MotionBlur)
            setattr(imaug_module, 'GaussianNoise', GaussianNoise)
            print("[+] Đăng ký thành công custom transforms (MotionBlur, GaussianNoise) vào ppocr.data.imaug")
        else:
            print("[!] Cảnh báo: Module ppocr.data.imaug chưa được tải. Hãy import ppocr trước.")
    except Exception as e:
        print(f"[!] Lỗi khi đăng ký custom transforms: {e}")
