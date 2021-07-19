import cv2 as cv
import matplotlib.pyplot as plt
import json
import os
import time  
from external_lib.Vi_cA_13 import Ring_Processer
from lib.core import Word_Classification

"""
获取一张图的半径，圆心信息
"""
def get_radius_center(img):
    ring_obj = Ring_Processer(img)
    circles = ring_obj.circle_list
    r_inner = circles[2][2]
    r_outer = circles[3][2]
    center = ring_obj.get_center(circles)
    return r_inner,r_outer,center

def main(img,bbox_list,r_inner,r_outer,center,pattern_list):
    distribution_classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    weights_path = "./assets/resnet34.pt"
    start = time.time()

    word_classifier = Word_Classification(weights_path,distribution_classes)
    is_NG, result = word_classifier.get_str_matchInfo(img,bbox_list,r_inner,r_outer,center,pattern_list)
    
    end = time.time()

    print("last: ",end-start)
    print("is_NG:",is_NG)
    print("info:",result)
    for xyxy in result["str_bbox_list"]:
        pt1 = (int(xyxy[0]),int(xyxy[1]))
        pt2 = (int(xyxy[2]),int(xyxy[3]))
        img = cv.rectangle(img, pt1, pt2, (0,0,255), thickness=10)
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.show()

if __name__=="__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    #load data
    img = cv.imread("./assets/test.png")
    img_json_path = "./assets/test.json"

    #get info
    r_inner,r_outer,center = get_radius_center(img)
    bbox_list = []
    with open(img_json_path,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    for item in json_data["shapes"]:
        if item["label"]=="word":
            bbox_list.append(item["points"])
    pattern_list = ["6202/P6","B"] 

    main(img,bbox_list,r_inner,r_outer,center,pattern_list)
