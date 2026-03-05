from torchvision.transforms import RandAugment
from math import sqrt

from .transforms import *

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


def get_augmentation(training: bool, config):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    input_size = config.data.input_size if config is not None else 224
    scale_size = input_size * 256 // 224 # scale_size / input_size = 256 / 224
    if training:
        unique = torchvision.transforms.Compose(
            [
                #GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                GroupScale(scale_size),
                GroupCenterCrop(input_size), # center crop seems more resonable than random crop
                GroupRandomHorizontalFlip(is_sth=False),
                GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                GroupRandomGrayscale(p=0.2),
                GroupGaussianBlur(p=0.0),
                GroupSolarization(p=0.0)
            ]
        )
    else:
        unique = torchvision.transforms.Compose(
            [
                GroupScale(scale_size),
                GroupCenterCrop(input_size)
            ]
        )
    common = torchvision.transforms.Compose(
        [
            Stack(roll=False), # ndarray: (224, 224, T*3)
            ToTorchFormatTensor(div=True), # tensor: (224, 224, T*3) -> (T*3, 224, 224)
            GroupNormalize(input_mean, input_std)
        ]
    )
    return torchvision.transforms.Compose([unique, common])


def rand_augment(transform, config): # add a random augmentation (implemented by torchvision) before the original transform
    transform.transforms.insert(0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M)))
    return transform


def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """
    Original version from "Causality Inspired Representation Learning for Domain Generalization" (CVPR '22)
    Input image size: ndarray of [H, W, C]
    """
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha)) # use phase from img1 and magnitude from img2
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12


def colorful_spectrum_mix(img1, img2, alpha):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)
    assert img1.shape == img2.shape

    x1 = img1.astype(np.float32, copy=False)
    x2 = img2.astype(np.float32, copy=False)

    F1 = np.fft.fft2(x1, axes=(0, 1))
    F2 = np.fft.fft2(x2, axes=(0, 1))

    pha1 = np.angle(F1) # phase from img1
    mag1 = np.abs(F1) # magnitude from img1
    mag2 = np.abs(F2) # magnitude from img2

    mag = (1.0 - lam) * mag1 + lam * mag2 # magnitude mix

    out = np.fft.ifft2(mag * np.exp(1j * pha1), axes=(0, 1)).real
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out