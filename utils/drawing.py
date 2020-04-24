import math

import cv2
import numpy as np


def draw_sol_torch(predictions, org_img, conf_threshold=0.1):
    h, w = org_img.shape[:2]
    org_img = cv2.resize(org_img, (w, h))
    for j in range(predictions.size(1)):

        conf = predictions[0, j, 0]
        conf = conf.item()
        color = int(255 * conf)

        pt0 = predictions[0, j, 1:3]  # * 512
        pt1 = predictions[0, j, 3:5]  # * 512

        pt0 = tuple(pt0.data.cpu().numpy().astype(np.int64).tolist())
        pt1 = tuple(pt1.data.cpu().numpy().astype(np.int64).tolist())

        x0, y0 = pt0
        x1, y1 = pt1

        dx = x0 - x1
        dy = y0 - y1

        d = math.sqrt(dx ** 2 + dy ** 2)

        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0

        scale = (d / 2)

        if conf < conf_threshold:
            continue

        mx = int(mx)
        my = int(my)
        scale = int(max(scale, 1))

        cv2.circle(org_img, (mx, my), scale, color, 1)
    return org_img
