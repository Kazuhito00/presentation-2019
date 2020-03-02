#!/usr/bin/env python
# -*- coding: utf-8 -*-

import eel
import cv2 as cv
import tensorflow as tf
import numpy as np
import base64
import copy

from CvPutJaText import CvPutJaText
from CvOverlayImage import CvOverlayImage
import FpsCalc


def session_run(sess, inp):
    out = sess.run([
        sess.graph.get_tensor_by_name('num_detections:0'),
        sess.graph.get_tensor_by_name('detection_scores:0'),
        sess.graph.get_tensor_by_name('detection_boxes:0'),
        sess.graph.get_tensor_by_name('detection_classes:0')
    ],
                   feed_dict={
                       'image_tensor:0':
                       inp.reshape(1, inp.shape[0], inp.shape[1], 3)
                   })
    return out


cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# フォント準備 ############################################################
font_path = './font/x12y20pxScanLine.ttf'
cvPutJaText = CvPutJaText()

# GPUメモリを必要な分だけ確保
# ※指定しない限りデフォルトではすべて確保する
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

# 手検出モデルロード #######################################################
with tf.Graph().as_default() as net1_graph:
    graph_data = tf.gfile.FastGFile('model/frozen_inference_graph.pb',
                                    'rb').read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(graph_data)
    tf.import_graph_def(graph_def, name='')

sess1 = tf.Session(graph=net1_graph, config=config)
sess1.graph.as_default()

# 初回実行に時間がかかるため、一度実施する
temp_inp = np.zeros((300, 300, 3), np.uint8)
session_run(sess1, temp_inp)

# FPS算出クラス起動 ######################################################
fpsWithTick = FpsCalc.fpsWithTick()
detection_fps = 0.0

# Set web files folder
eel.init('web')

eel.start(
    'index.html',
    mode='chrome',
    # cmdline_args=['--start-fullscreen', '--browser-startup-dialog'])
    cmdline_args=['--start-fullscreen'],
    block=False)

while True:
    eel.sleep(0.01)

    # FPS算出 ###########################################################
    display_fps = fpsWithTick.get()
    if display_fps == 0:
        display_fps = 0.1

    # カメラキャプチャ ###################################################
    ret, frame = cap.read()
    if not ret:
        continue

    # 検出実施 ####################################################
    analysis_image = copy.deepcopy(frame)
    inp = cv.resize(analysis_image, (512, 512))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    out = session_run(sess1, inp)

    rows = frame.shape[0]
    cols = frame.shape[1]

    # 検出結果可視化 ###############################################
    num_detections = int(out[0][0])
    for i in range(num_detections):
        class_id = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]

        if score < 0.8:
            continue

        x = bbox[1] * cols
        y = bbox[0] * rows
        right = bbox[3] * cols
        bottom = bbox[2] * rows

        if class_id == 1:
            cv.rectangle(
                frame, (int(x), int(y)), (int(right), int(bottom)),
                (255, 255, 205),
                thickness=10)
            frame = cvPutJaText.puttext(frame, u"Open",
                                        (int(x) + 50, int(y) + 50), font_path,
                                        80, (205, 255, 255))

        elif class_id == 2:
            cv.circle(frame, (int((x + right) / 2), int((y + bottom) / 2)),
                      int((right - x) * (3 / 5)), (255, 255, 205), 10)
            frame = cvPutJaText.puttext(frame, u"Close",
                                        (int(x) + 50, int(y) + 50), font_path,
                                        80, (205, 255, 255))

        elif class_id == 3:
            trimming_image = frame[int(y):int(bottom), int(x):int(right)]

            # cv.imwrite(
            #     '.\\output\\frame{:04}.png'.format(
            #         write_count), trimming_image)
            # write_count += 1

            pts = np.array([
                [int(x + ((right - x) * (1 / 5))),
                 int(y)],
                [int(x + ((right - x) * (4 / 5))),
                 int(y)],
                [int(right), int(y + ((bottom - y) * (1 / 5)))],
                [int(right), int(y + ((bottom - y) * (4 / 5)))],
                [int(x + ((right - x) * (4 / 5))),
                 int(bottom)],
                [int(x + ((right - x) * (1 / 5))),
                 int(bottom)],
                [int(x), int(y + ((bottom - y) * (4 / 5)))],
                [int(x), int(y + ((bottom - y) * (1 / 5)))],
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            frame = cv.polylines(
                frame, [pts], True, (255, 255, 205), thickness=10)
            frame = cvPutJaText.puttext(frame, u"Pointer",
                                        (int(x) + 50, int(y) + 50), font_path,
                                        80, (205, 255, 255))

        frame = cvPutJaText.puttext(frame, '{:.5g}'.format(score),
                                    (int(x) + 50, int(y) + 130), font_path, 40,
                                    (205, 255, 255))

    # FPS描画
    fps_string = "FPS:" + str(display_fps)
    image = cvPutJaText.puttext(frame, fps_string, (int(10), int(10)),
                                font_path, 36, (0, 255, 0))
    # UI側へ転送
    _, imencode_image = cv.imencode('.jpg', image)
    base64_image = base64.b64encode(imencode_image)
    eel.demo01_set_base64image("data:image/jpg;base64," +
                               base64_image.decode("ascii"))

    # cv.imshow('webslides debug', image)

    key = cv.waitKey(10)
    if key == 27:  # ESC
        break
