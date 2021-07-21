
import sys
import cv2 as cv
import matplotlib.pyplot as plt
sys.path.append('external_lib/mmdetection')
import json
import os
import time
from external_lib.Vi_cA_13 import Ring_Processer
from lib.core import Word_Classification


def main(img, circles, pattern_list):

    start = time.time()
    word_classifier = Word_Classification(gpu_id=0)
    is_NG, result = word_classifier.get_str_matchInfo(img, circles, pattern_list)
    end = time.time()

    print("last: ", end-start)
    print("is_NG:", is_NG)
    print("info:", result)
    for xyxy in result["str_bbox_list"]:
        pt1 = (int(xyxy[0]), int(xyxy[1]))
        pt2 = (int(xyxy[2]), int(xyxy[3]))
        img = cv.rectangle(img, pt1, pt2, (0, 0, 255), thickness=10)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # load data
    img = cv.imread("./assets/test.png")

    # get info
    circles = Ring_Processer(img).circle_list

    pattern_list = ["6202/P6", "BH", 'A9607A4-98']

    main(img, circles, pattern_list)
