# coding utf-8
import os
import os.path as osp
import cv2
import numpy as np
import glob
from PIL import Image, ImageFont, ImageDraw
import random


def watermark(image, text, size=None, color=None, alpha=1.0, position=0):
    back = Image.fromarray(image).convert('RGBA') if type(image) is np.ndarray else image.convert('RGBA')
    fore = Image.new(back.mode, back.size, (0, 0, 0, 0))
    # size = min(back.width, back.height) // 20 if size is None else max(20, size)
    size_offset = random.randint(0,3) * 0.001
    size += size_offset
    size = max(20, min(int(back.width * size), int(back.height * size), 30))
    font = ImageFont.truetype("arial.ttf", size)
    w, h = font.getsize(text)
    rgba = (255, 255, 255) if color is None else color
    rgba = rgba + (int(255 * alpha),)
    if position == 0:
        x, y = (back.width - w) // 2, (back.height - h) // 2
    elif position == 1:
        x, y = 8, 4
    elif position == 2:
        x, y = back.width - w - 8, 4
    elif position == 3:
        offset = random.randint(8,10)
        x, y = back.width - w - offset, back.height - h - offset
    elif position == 4:
        x, y = 8, back.height - h - 8
    draw = ImageDraw.Draw(fore)
    draw.text((x, y), text, rgba, font)
    output = Image.alpha_composite(back, fore).convert('RGB')
    return np.uint8(output) if type(image) is np.ndarray else output

def img_mark(input_dir):
    img_list = glob.glob(input_dir + '/**/images/'+ '*.jpg',recursive=True)
    for img_path in img_list:
        retpath = img_path.replace('images', 'imgs')
        if osp.exists(retpath):
            continue
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        water_log = "github: Autopilot-Updating-Notes"
        image = watermark(image, water_log, alpha=1.0, color=(169,169,169), position=3, size=0.03)
        # cv2.imwrite(retpath, image)
        cv2.imencode('.jpg', image)[1].tofile(retpath)

input_dir = '.'
img_mark(input_dir)
