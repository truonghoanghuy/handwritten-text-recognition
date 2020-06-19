from typing import Tuple

import cv2
import numpy as np
from imgaug import augmenters as iaa

from hw import grid_distortion


def tensmeyer_brightness(img, foreground=0, background=0):
    ret, th = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th = (th.astype(np.float32) / 255)[..., None]

    img = img.astype(np.float32)
    img = img + (1.0 - th) * foreground
    img = img + th * background

    img[img > 255] = 255
    img[img < 0] = 0

    return img.astype(np.uint8)


def apply_tensmeyer_brightness(img, sigma=30, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    foreground = random_state.normal(0, sigma)
    background = random_state.normal(0, sigma)

    img = tensmeyer_brightness(img, foreground, background)

    return img


def increase_brightness(img, brightness=0, contrast=1):
    img = img.astype(np.float32)
    img = img * contrast + brightness
    img[img > 255] = 255
    img[img < 0] = 0

    return img.astype(np.uint8)


def apply_random_brightness(img, b_range=None, **kwargs):
    if b_range is None:
        b_range = [-50, 51]
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    brightness = random_state.randint(b_range[0], b_range[1])

    img = increase_brightness(img, brightness)

    return img


def apply_random_color_rotation(img, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    shift = random_state.randint(0, 255)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = hsv[..., 0] + shift
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


def otsu(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    thr = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
    return thr


def draw_background_grid(img: np.ndarray, grid_size: int = None, color: Tuple = None):
    randint = np.random.randint
    if not isinstance(img, np.ndarray):
        raise TypeError('Input image must be a numpy array')
    if not grid_size:
        grid_size = randint(8, 16)
    if not color:
        color = (randint(150, 256), randint(150, 256), randint(150, 256))

    img_height, img_width = img.shape[:2]
    background = np.ones(img.shape, dtype=np.uint8) * 255
    x0 = randint(-grid_size, 0)
    y0 = randint(-grid_size, 0)
    xn = randint(img_width, img_width + grid_size)
    yn = randint(img_height, img_height + grid_size)
    for x in range(x0, xn, grid_size):
        cv2.line(background, (x, y0), (x, yn), color=color)
    for y in range(y0, yn, grid_size):
        cv2.line(background, (x0, y), (xn, y), color=color)
    img = np.where(img > background, background, img)
    return img


class ImgAugAugmenter:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.SomeOf(n=(0, None), children=[
                iaa.ReplaceElementwise(mask=(0, 0.02), replacement=(0, 255), per_channel=0.5),
                iaa.OneOf([
                    iaa.Emboss(alpha=0.3, strength=0.3),
                    iaa.AddElementwise(value=(-10, 10), per_channel=0.5),
                    iaa.AdditiveGaussianNoise(scale=(0, 3), per_channel=0.5),
                    iaa.AdditiveLaplaceNoise(scale=(0, 3), per_channel=0.5),
                    iaa.AdditivePoissonNoise(lam=(0, 3), per_channel=0.5),
                ]),
            ]),
            iaa.ScaleX(scale=(0.8, 1.25), fit_output=True),
            iaa.Sometimes(0.5, iaa.JpegCompression(compression=10)),
            iaa.Invert(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class HwAugmenter:
    def __init__(self):
        self.img_aug_augmenter = ImgAugAugmenter()

    def __call__(self, img):
        if np.random.rand() < 0.8:
            img = apply_random_color_rotation(img)
            img = apply_tensmeyer_brightness(img)
            if np.random.rand() < 0.5:
                img = draw_background_grid(img)
            if np.random.rand() < 0.5:
                img = grid_distortion.warp_image(img)
            img = self.img_aug_augmenter(img)
        return img
