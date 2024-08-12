# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image, ImageEnhance  # type: ignore
from scipy.ndimage import distance_transform_edt  # type: ignore


def crop(img, crop_coordinate):
    x1, y1, x2, y2 = crop_coordinate
    img = img[y1:y2, x1:x2, ...]
    return img


def rescale_size(img_size, target_size):
    scale = min(max(target_size) / max(img_size), min(target_size) / min(img_size))
    rescaled_size = [round(i * scale) for i in img_size]
    return rescaled_size, scale


def normalize(im, mean, std):
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def resize(im, target_size=608, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


def resize_long(im, long_size=224, interpolation=cv2.INTER_LINEAR):
    value = max(im.shape[0], im.shape[1])
    scale = float(long_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(im, (resized_width, resized_height), interpolation=interpolation)
    return im


def resize_short(im, short_size=224, interpolation=cv2.INTER_LINEAR):
    value = min(im.shape[0], im.shape[1])
    scale = float(short_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(im, (resized_width, resized_height), interpolation=interpolation)
    return im


def horizontal_flip(im):
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    return im


def vertical_flip(im):
    if len(im.shape) == 3:
        im = im[::-1, :, :]
    elif len(im.shape) == 2:
        im = im[::-1, :]
    return im


def brightness(im, brightness_lower, brightness_upper):
    brightness_delta = np.random.uniform(brightness_lower, brightness_upper)
    im = ImageEnhance.Brightness(im).enhance(brightness_delta)
    return im


def contrast(im, contrast_lower, contrast_upper):
    contrast_delta = np.random.uniform(contrast_lower, contrast_upper)
    im = ImageEnhance.Contrast(im).enhance(contrast_delta)
    return im


def saturation(im, saturation_lower, saturation_upper):
    saturation_delta = np.random.uniform(saturation_lower, saturation_upper)
    im = ImageEnhance.Color(im).enhance(saturation_delta)
    return im


def hue(im, hue_lower, hue_upper):
    hue_delta = np.random.uniform(hue_lower, hue_upper)
    im = np.array(im.convert("HSV"))
    im[:, :, 0] = im[:, :, 0] + hue_delta
    im = Image.fromarray(im, mode="HSV").convert("RGB")
    return im


def sharpness(im, sharpness_lower, sharpness_upper):
    sharpness_delta = np.random.uniform(sharpness_lower, sharpness_upper)
    im = ImageEnhance.Sharpness(im).enhance(sharpness_delta)
    return im


def rotate(im, rotate_lower, rotate_upper):
    rotate_delta = np.random.uniform(rotate_lower, rotate_upper)
    im = im.rotate(int(rotate_delta))
    return im


def mask_to_onehot(mask, num_classes):
    """
    Convert a mask (H, W) to onehot (K, H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Onehot mask with shape(K, H, W).
    """
    _mask = [mask == i for i in range(num_classes)]
    _mask = np.array(_mask).astype(np.uint8)
    return _mask


def onehot_to_binary_edge(mask, radius):
    """
    Convert a onehot mask (K, H, W) to a edge mask.

    Args:
        mask (np.ndarray): Onehot mask with shape (K, H, W)
        radius (int|float): Radius of edge.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    if radius < 1:
        raise ValueError("`radius` should be greater than or equal to 1")
    num_classes = mask.shape[0]

    edge = np.zeros(mask.shape[1:])
    # pad borders
    mask = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
    for i in range(num_classes):
        dist = distance_transform_edt(mask[i, :]) + distance_transform_edt(
            1.0 - mask[i, :]
        )
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edge += dist

    edge = np.expand_dims(edge, axis=0)
    edge = (edge > 0).astype(np.uint8)
    return edge


def mask_to_binary_edge(mask, radius, num_classes):
    """
    Convert a segmentic segmentation mask (H, W) to a binary edge mask(H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        radius (int|float): Radius of edge.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    mask = mask.squeeze()
    onehot = mask_to_onehot(mask, num_classes)
    edge = onehot_to_binary_edge(onehot, radius)
    return edge


# TIFF-specific functions
def gaussian_noise(image: np.ndarray, noise_scale: float = 2.0) -> np.ndarray:
    """
    Add Gaussian noise to a TIFF image.

    Args:
        image (np.ndarray): The image to which Gaussian noise will be added.
        noise_scale (float): The maximum noise scaling value (default: 1.0).

    Returns:
        np.ndarray: The image after adding Gaussian noise.
    """
    #   get image statistics
    try:
        dtype = np.finfo(image.dtype)
    except ValueError:
        #  image is not a float
        dtype = np.iinfo(image.dtype)
    min_val, max_val = dtype.min, dtype.max

    std = np.std(image) * np.random.uniform(low=0, high=noise_scale)
    noise = np.random.normal(0, std, image.shape)
    image = image + noise

    image = np.clip(image, min_val, max_val).astype(dtype)
    del dtype, min_val, max_val, std, noise

    return image


def gaussian_blur(image):
    """
    Gaussian blur a TIFF image.

    Args:
        image (np.ndarray): The image to which Gaussian blur will be applied.

    Returns:
        np.ndarray: The image after applying Gaussian blur.

    """
    sigma = np.random.uniform(low=0, high=1)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    del sigma

    return image


def pixel_dropout(image):
    """
    Randomly replace 1–10% of TIFF image pixels with the image mean value.

    Args:
        image (np.ndarray): The image to which pixel dropout will be applied.

    Returns:
        np.ndarray: The image after applying pixel dropout.
    """
    #   determine pixels to be set to image mean
    h, w, _ = image.shape
    total = h * w
    pixels = np.rint(total * (np.random.randint(1, 11) / 100)).astype(np.uint32)
    pixels = np.random.choice(total, size=pixels, replace=False)
    rows, cols = (pixels // w) - 1, (pixels % w) - 1

    #   replace select pixels with image mean
    image[rows, cols] = np.mean(image)
    del h, w, _, total, pixels, rows, cols

    return image


def contrast_stretching(image):
    """
    Apply contrast stretching to an image.

    Args:

    """
    #   get image statistics
    min_val = np.min(image)
    max_val = np.max(image)
    median = np.median(image)

    #   apply contrast stretching
    image = ((image - median) * np.random.uniform(low=0, high=1)) + median
    image = np.clip(image, min_val, max_val)
    del min_val, max_val, median

    return image
