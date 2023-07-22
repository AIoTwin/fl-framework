import logging
import os
from logging import FileHandler, Formatter
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOGGING_FORMAT = "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"

logging.basicConfig(
    format=LOGGING_FORMAT,
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

def_logger = logging.getLogger()


def make_parent_dirs(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def uniquify(path) -> str:
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def _setup_log_file(log_file_path, mode="w"):
    make_parent_dirs(log_file_path)
    fh = FileHandler(filename=log_file_path, mode=mode)
    fh.setFormatter(Formatter(LOGGING_FORMAT))
    def_logger.addHandler(fh)


def prepare_local_log_file(
        log_file_path,
        # config_path,
        mode: str = "w",
        overwrite=False,
        test_only=False

):
    eval_file = "_eval" if test_only else ""
    # if log_file_path:
    #     log_file_path = (
    #         f"{os.path.join(log_file_path, Path(config_path).stem)}{eval_file}.log"
    #     )
    # else:
    #     log_file_path = (
    #         f"{config_path.replace('config', 'logs', 1)}{eval_file}".replace(
    #             ".yaml", ".log", 1
    #         )
    #     )
    if mode == "w" and not overwrite:
        log_file_path = uniquify(log_file_path)
    _setup_log_file(os.path.expanduser(log_file_path), mode=mode)


def show_att_map_on_image(
        img: np.ndarray,
        mask: np.ndarray,
        use_rgb: bool = False,
        colormap: int = cv2.COLORMAP_JET,
        image_weight: float = 0.5,
) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}"
        )

    map = (1 - image_weight) * heatmap + image_weight * img
    map = map / np.max(map)
    return np.uint8(255 * map)


def concat_images_h(
        img_a: Image.Image, img_b: Image.Image, margin: Optional[int] = None
) -> Image.Image:
    dst = Image.new("RGB", (img_b.width + img_a.width, img_b.height))
    dst.paste(img_b, (0, 0))
    dst.paste(img_a, (img_b.width, 0))
    return dst


def concat_images_h_caption_metric(
        img_a: Image.Image,
        img_b: Tuple[Image.Image, Optional[str]],
        metric: Optional[str] = None,
) -> Image.Image:
    img_b, metric_val = img_b
    if metric:
        recon_img_cap = ImageDraw.Draw(img_b)
        recon_img_cap.text(
            (0, 0),
            f"{metric}={metric_val}",
            font=ImageFont.truetype("FreeMono.ttf", 30),
            fill=(0, 0, 0),
        )
    dst = Image.new("RGB", (img_b.width + img_a.width, img_b.height))
    dst.paste(img_b, (5, 5))
    dst.paste(img_a, (img_b.width, 0))
    return dst


def concat_images_v(
        img_a: Image.Image, img_b: Image.Image, margin: Optional[int] = None
) -> Image.Image:
    dst = Image.new("RGB", (img_b.width, img_b.height + img_a.height))
    dst.paste(img_b, (0, 0))
    dst.paste(img_a, (0, img_b.height))
    return dst
