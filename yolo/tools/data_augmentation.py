from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

import math
import random
import albumentations as A


class AugmentationComposer:
    """Composes several transforms together."""

    def __init__(self, transforms, image_size: int = [640, 640], base_size: int = 640):
        self.transforms = transforms
        # TODO: handle List of image_size [640, 640]
        self.pad_resize = PadAndResize(image_size)
        self.base_size = base_size

        for transform in self.transforms:
            if hasattr(transform, "set_parent"):
                transform.set_parent(self)

    def __call__(self, image, boxes=torch.zeros(0, 5)):
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        image, boxes, rev_tensor = self.pad_resize(image, boxes)
        image = TF.to_tensor(image)
        return image, boxes, rev_tensor


class RandomSunFlare(A.DualTransform):
    """Simulates Sun Flare for the image
    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Modified from official Albumentations implementation to restrict the flare
    location within the target's bounding box

    Args:
        flare_roi (float, float, float, float): region of the image where flare will
            appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].
        angle_lower (float): should be in range [0, `angle_upper`].
        angle_upper (float): should be in range [`angle_lower`, 1].
        num_flare_circles_lower (int): lower limit for the number of flare circles.
            Should be in range [0, `num_flare_circles_upper`].
        num_flare_circles_upper (int): upper limit for the number of flare circles.
            Should be in range [`num_flare_circles_lower`, inf].
        src_radius (int):
        src_color ((int, int, int)): color of the flare
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        prob=0.5,
        angle_lower=0,
        angle_upper=1,
        num_flare_circles_lower=1,
        num_flare_circles_upper=10,
        always_apply=False
    ):
        super(RandomSunFlare, self).__init__(always_apply, prob)

        if not 0 <= angle_lower < angle_upper <= 1:
            raise ValueError(
                "Invalid combination of angle_lower nad angle_upper. Got: {}".format((angle_lower, angle_upper))
            )
        if not 0 <= num_flare_circles_lower < num_flare_circles_upper:
            raise ValueError(
                "Invalid combination of num_flare_circles_lower nad num_flare_circles_upper. Got: {}".format(
                    (num_flare_circles_lower, num_flare_circles_upper)
                )
            )

        self.angle_lower = angle_lower
        self.angle_upper = angle_upper
        self.num_flare_circles_lower = num_flare_circles_lower
        self.num_flare_circles_upper = num_flare_circles_upper
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            params = self.get_params_dependent_on_targets(image,boxes)
            self.apply(image,params['src_radius'],params['flare_center_x'],params['flare_center_y'],params['circles'])
        return image, boxes

    def apply(self, image, src_radius=400, flare_center_x=0.5, flare_center_y=0.5, circles=(), **params):
        clr = random.choice([0, 255])
        src_color = (clr, clr, clr)
        image_np = A.augmentations.pixel.functional.add_sun_flare_overlay(np.array(image), (flare_center_x,flare_center_y), src_radius, src_color, circles)
        image = Image.fromarray(image_np)
        return image

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_params_dependent_on_targets(self, image, boxes):
        img = np.array(image)
        bbox = boxes[0] # Single object [xmin, ymin, xmax, ymax] (normalized)
        height, width = img.shape[:2]

        # Flare angle
        angle = 2 * math.pi * random.uniform(self.angle_lower, self.angle_upper)

        # Flare location
        flare_center_x = random.uniform(bbox[0], bbox[2])
        flare_center_y = random.uniform(bbox[1], bbox[3])

        flare_center_x = int(width * flare_center_x)
        flare_center_y = int(height * flare_center_y)

        # Number of circles
        num_circles = random.randint(self.num_flare_circles_lower, self.num_flare_circles_upper)

        # Random color
        src_color = random.randint(0, 256)

        # Random radius smaller than target
        src_radius = int(random.uniform(0.2, 0.4) * max(width, height))

        # Create circles
        circles = []

        x = []
        y = []

        for rand_x in range(0, width, 10):
            rand_y = math.tan(angle) * (rand_x - flare_center_x) + flare_center_y
            x.append(rand_x)
            y.append(2 * flare_center_y - rand_y)

        for _i in range(num_circles):
            alpha = random.uniform(0.05, 0.2)
            r = random.randint(0, len(x) - 1)
            rad = random.randint(1, max(height // 100 - 2, 2))

            r_color = random.randint(max(src_color - 50, 0), src_color)
            g_color = random.randint(max(src_color - 50, 0), src_color)
            b_color = random.randint(max(src_color - 50, 0), src_color)

            circles += [
                (
                    alpha,
                    (int(x[r]), int(y[r])),
                    pow(rad, 3),
                    (r_color, g_color, b_color),
                )
            ]

        return {
            "circles": circles,
            "src_radius": src_radius,
            "flare_center_x": flare_center_x,
            "flare_center_y": flare_center_y,
        }

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_transform_init_args(self):
        return {
            "angle_lower": self.angle_lower,
            "angle_upper": self.angle_upper,
            "num_flare_circles_lower": self.num_flare_circles_lower,
            "num_flare_circles_upper": self.num_flare_circles_upper,
        }
#######################

class RandomBrightness:
    def __init__(self, prob=0.5, factor_range=[0.8, 1.2]):
        self.prob = prob
        self.factor_range = factor_range

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            factor = np.random.uniform(self.factor_range[0], self.factor_range[1])
            image = TF.adjust_brightness(image, factor)
        return image, boxes


class RandomContrast:
    def __init__(self, prob=0.5, factor_range=[0.8, 1.2]):
        self.prob = prob
        self.factor_range = factor_range

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            factor = np.random.uniform(self.factor_range[0], self.factor_range[1])
            image = TF.adjust_contrast(image, factor)
        return image, boxes


class GaussianBlur:
    def __init__(self, prob=0.5, kernel_size=5):
        self.prob = prob
        self.kernel_size = kernel_size

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.gaussian_blur(image, self.kernel_size)
        return image, boxes


class GaussianNoise:
    def __init__(self, prob=0.5, std=5):
        self.prob = prob
        self.std = std

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image_np = np.array(image)
            noise = self.std*np.random.randn(*image_np.shape)
            image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_np)
        return image, boxes


class RemoveOutliers:
    """Removes outlier bounding boxes that are too small or have invalid dimensions."""

    def __init__(self, min_box_area=1e-8):
        """
        Args:
            min_box_area (float): Minimum area for a box to be kept, as a fraction of the image area.
        """
        self.min_box_area = min_box_area

    def __call__(self, image, boxes):
        """
        Args:
            image (PIL.Image): The cropped image.
            boxes (torch.Tensor): Bounding boxes in normalized coordinates (x_min, y_min, x_max, y_max).
        Returns:
            PIL.Image: The input image (unchanged).
            torch.Tensor: Filtered bounding boxes.
        """
        box_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])

        valid_boxes = (box_areas > self.min_box_area) & (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 4] > boxes[:, 2])

        return image, boxes[valid_boxes]


class PadAndResize:
    def __init__(self, image_size, background_color=(114, 114, 114)):
        """Initialize the object with the target image size."""
        self.target_width, self.target_height = image_size
        self.background_color = background_color

    def set_size(self, image_size: List[int]):
        self.target_width, self.target_height = image_size

    def __call__(self, image: Image, boxes):
        img_width, img_height = image.size
        scale = min(self.target_width / img_width, self.target_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        pad_left = (self.target_width - new_width) // 2
        pad_top = (self.target_height - new_height) // 2
        padded_image = Image.new("RGB", (self.target_width, self.target_height), self.background_color)
        padded_image.paste(resized_image, (pad_left, pad_top))

        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * new_width + pad_left) / self.target_width
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] * new_height + pad_top) / self.target_height

        transform_info = torch.tensor([scale, pad_left, pad_top, pad_left, pad_top])
        return padded_image, boxes, transform_info


class HorizontalFlip:
    """Randomly horizontally flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.hflip(image)
            boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]
        return image, boxes


class VerticalFlip:
    """Randomly vertically flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.vflip(image)
            boxes[:, [2, 4]] = 1 - boxes[:, [4, 2]]
        return image, boxes


class Mosaic:
    """Applies the Mosaic augmentation to a batch of images and their corresponding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        assert self.parent is not None, "Parent is not set. Mosaic cannot retrieve image size."

        img_sz = self.parent.base_size  # Assuming `image_size` is defined in parent
        more_data = self.parent.get_more_data(3)  # get 3 more images randomly

        data = [(image, boxes)] + more_data
        mosaic_image = Image.new("RGB", (2 * img_sz, 2 * img_sz), (114, 114, 114))
        vectors = np.array([(-1, -1), (0, -1), (-1, 0), (0, 0)])
        center = np.array([img_sz, img_sz])
        all_labels = []

        for (image, boxes), vector in zip(data, vectors):
            this_w, this_h = image.size
            coord = tuple(center + vector * np.array([this_w, this_h]))

            mosaic_image.paste(image, coord)
            xmin, ymin, xmax, ymax = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
            xmin = (xmin * this_w + coord[0]) / (2 * img_sz)
            xmax = (xmax * this_w + coord[0]) / (2 * img_sz)
            ymin = (ymin * this_h + coord[1]) / (2 * img_sz)
            ymax = (ymax * this_h + coord[1]) / (2 * img_sz)

            adjusted_boxes = torch.stack([boxes[:, 0], xmin, ymin, xmax, ymax], dim=1)
            all_labels.append(adjusted_boxes)

        all_labels = torch.cat(all_labels, dim=0)
        mosaic_image = mosaic_image.resize((img_sz, img_sz))
        return mosaic_image, all_labels


class MixUp:
    """Applies the MixUp augmentation to a pair of images and their corresponding boxes."""

    def __init__(self, prob=0.5, alpha=1.0):
        self.alpha = alpha
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        """Set the parent dataset object for accessing dataset methods."""
        self.parent = parent

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        assert self.parent is not None, "Parent is not set. MixUp cannot retrieve additional data."

        # Retrieve another image and its boxes randomly from the dataset
        image2, boxes2 = self.parent.get_more_data()[0]

        # Calculate the mixup lambda parameter
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 0.5

        # Mix images
        image1, image2 = TF.to_tensor(image), TF.to_tensor(image2)
        mixed_image = lam * image1 + (1 - lam) * image2

        # Merge bounding boxes
        merged_boxes = torch.cat((boxes, boxes2))

        return TF.to_pil_image(mixed_image), merged_boxes


class RandomCrop:
    """Randomly crops the image to half its size along with adjusting the bounding boxes."""

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): Probability of applying the crop.
        """
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            original_width, original_height = image.size
            crop_height, crop_width = original_height // 2, original_width // 2
            top = torch.randint(0, original_height - crop_height + 1, (1,)).item()
            left = torch.randint(0, original_width - crop_width + 1, (1,)).item()

            image = TF.crop(image, top, left, crop_height, crop_width)

            boxes[:, [1, 3]] = boxes[:, [1, 3]] * original_width - left
            boxes[:, [2, 4]] = boxes[:, [2, 4]] * original_height - top

            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, crop_width)
            boxes[:, [2, 4]] = boxes[:, [2, 4]].clamp(0, crop_height)

            boxes[:, [1, 3]] /= crop_width
            boxes[:, [2, 4]] /= crop_height

        return image, boxes
