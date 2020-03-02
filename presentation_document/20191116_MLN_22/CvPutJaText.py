#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from PIL import ImageFont, ImageDraw, Image

class CvPutJaText:
    def __init__(self):
        pass

    @classmethod
    def puttext(cls, cv_image, text, point, font_path, font_size, color=(0,0,0)):
        font = ImageFont.truetype(font_path, font_size)
        
        cv_rgb_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_rgb_image)
        
        draw = ImageDraw.Draw(pil_image)
        draw.text(point, text, fill=color, font=font)
        
        cv_rgb_result_image = np.asarray(pil_image)
        cv_bgr_result_image = cv.cvtColor(cv_rgb_result_image, cv.COLOR_RGB2BGR)

        return cv_bgr_result_image

if __name__ == '__main__':
    cv_image = cv.imread("sample.jpg")
    
    font_path = './font/font_jb004_running_brush_wi.ttf'
    
    image = CvPutJaText.puttext(cv_image, u"ごんべ", (30, 30), font_path, 60, (0, 0, 0))

    cv.imshow("sample", image)
    cv.waitKey(0)
