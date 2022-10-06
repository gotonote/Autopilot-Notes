# coding utf-8
import os
import os.path as osp
import cv2
import numpy
from PIL import Image, ImageFont, ImageDraw


def watermark(image, text, size=None, color=None, alpha=1.0, position=0):
    back = Image.fromarray(image).convert('RGBA') if type(image) is numpy.ndarray else image.convert('RGBA')
    fore = Image.new(back.mode, back.size, (0, 0, 0, 0))
    # size = min(back.width, back.height) // 20 if size is None else max(20, size)
    size = min(int(back.width * size), int(back.height * size)) 
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
        x, y = back.width - w - 8, back.height - h - 8
    elif position == 4:
        x, y = 8, back.height - h - 8
    draw = ImageDraw.Draw(fore)
    draw.text((x, y), text, rgba, font)
    output = Image.alpha_composite(back, fore).convert('RGB')
    return numpy.uint8(output) if type(image) is numpy.ndarray else output

def img_mark(input_dir, output_dir):
    img_list = os.listdir(input_dir)
    img_list = [osp.join(input_dir, i) for i in img_list if '.jpg' in i]

    print(img_list)
    for img_path in img_list:
        img_name = osp.basename(img_path)
        retpath = osp.join(output_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        water_log = "github: Autopilot-Updating-Notes"
        image = watermark(image, water_log, alpha=0.9, color=(169,169,169), position=3, size=0.02)
        cv2.imwrite(retpath, image)

input_dir = './images'
output_dir = './imgs'
img_mark(input_dir, output_dir)
