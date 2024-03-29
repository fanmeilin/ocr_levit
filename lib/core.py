from .classifier import Controller as ClsController
from .detector import ocrA_13_det as DetController
import cmath
import math
import cv2 as cv
import re
import numpy as np
from .detector.lib.utils_vvd import boxes_painter
from .detector.lib.utils_vvd import plt_image_show


class Word_Classification:
    def __init__(self, gpu_id=0):
        self.cc_obj = ClsController(gpu_id=gpu_id)
        self.det_obj = DetController(gpu_id=gpu_id)

    @staticmethod
    def get_img_character_position(bbox_list, center):
        '''
        得到一张的所有字符的相对位置
        一个字符的位置表示 在图中的中心位置坐标 相对圆心的角度 长度（角度为角度值）
        input：所有字符的box 圆中心坐标center
        return 所有字符的位置信息
        '''

        def getCharacterCenterByInfo(Info):
            '''
            得到字符的中心点
            输入：相关最大最小值信息
            输出：中心点坐标
            '''
            result = [(Info[0]+Info[2])/2, (Info[1]+Info[3])/2]
            return result

        img_character_infolist = []
        for item in bbox_list:
            xyxy = item[0]+item[1]
            CharacterCenter = getCharacterCenterByInfo(xyxy)
            xy = (CharacterCenter[0]-center[0], center[1] -
                  CharacterCenter[1])  # 得到字符中心点相对圆心的直角坐标
            cn = complex(xy[0], xy[1])
            r, angle = cmath.polar(cn)  # 返回长度和弧度
    #         if(angle<0):
    #             angle += 2*math.pi
            angle = angle/math.pi*180
            img_character_infolist.append([CharacterCenter, angle, r, xyxy])
        return img_character_infolist

    @staticmethod
    def cluster_character(img_position, radius, thresh=185):
        """
        通过img的位置信息进行聚类 聚类的角度标准为thresh 
        input: img_position保存字符【【[x,y],xita,r】,【...】】 中心点 角度 半径的信息 thread:夹角小于thresh判定为一类
        return:聚类后的列表
        """

        def index_next(index, length, next_index=True):
            """
            list 实际上应该是一个首尾相连的结构
            可以指定index 找到其下一个或者上一个元素的下标
            input: 下标 列表长度 是否找寻下一个 T则找寻下一个 F则找寻上一个
            return：返回指定元素的下/上元素
            """
            if (next_index):
                index_another = index+1
                if(index_another == length):
                    index_another = 0
            if not(next_index):
                index_another = index-1
                if(index_another == -1):
                    index_another = length-1

            return index_another

        def inter_angle(img_position, index, next_index=True):
            """
            找寻指定的上，下夹角 
            input: 保存角度的list index  next_index
            return: next_index为真则返回与顺时针的相邻元素的夹角 为反则是返回逆时针夹角
            """
            index_another = index_next(index, len(img_position), next_index)
            angle = abs(img_position[index][1]-img_position[index_another][1])
            if(angle > 180):
                angle = 360-angle
            return angle

        def get_arc_length(angle, radius):
            """
            得到两个字符之间的弧长
            """
            return radius*(angle/180*math.pi)
        if(len(img_position) == 1):
            return [img_position]

        img_position = sorted(img_position, key=lambda x: (x[1]), reverse=True)
        start_index = 0
        length = len(img_position)
        result_list = []
        if(len(img_position) > 2):
            while(get_arc_length(inter_angle(img_position, start_index, False), radius) < thresh):  # 找寻开头元素
                start_index = index_next(start_index, length, False)

        select_index = index_next(start_index, length)  # 待聚类的元素
        result = [img_position[start_index]]
        while(True):
            if(get_arc_length(inter_angle(img_position, select_index, False), radius) < thresh):  # 小于指定角度则添加
                result.append(img_position[select_index])
            else:
                result_list.append(result)  # 否则此区域结束
                result = [img_position[select_index]]
            select_index = index_next(select_index, length)
            if(select_index == start_index):
                result_list.append(result)
                break
        return result_list

    @staticmethod
    def find_strbbox(cluster_character_list):
        """find img cluster bbox"""
        str_bbox_list = []
        for cluster in cluster_character_list:
            xmin = min(cluster, key=lambda x: x[3][0])[3][0]
            ymin = min(cluster, key=lambda x: x[3][1])[3][1]
            xmax = max(cluster, key=lambda x: x[3][2])[3][2]
            ymax = max(cluster, key=lambda x: x[3][3])[3][3]
            str_bbox = list(map(lambda x: round(x), [xmin, ymin, xmax, ymax]))
            str_bbox_list.append(str_bbox)
        return str_bbox_list

    @staticmethod
    def getPatternByCenter(img, circleCenter, characterCenter, Rlength, pad=0, boderValue=0):
        '''
        通过圆心，字符中心点得到切割矫正后的图片
        输入：圆心坐标：[x,y],多边形框的顶点位置：[[x,y],...],旋转后填充的长宽像素值 Rlength=剪裁的边长
        输出：剪裁矫正后的字符图像
        '''

        def calcAngle(point, center):
            '''
            计算绕中心点旋转角度 顺时针为负 逆时针为正
            输入：中心点的[x,y] 圆心坐标[x,y]
            输出：旋转度数
            '''
            angle = 0  # 角度为负，顺时针；角度为正，逆时针
            xdis = center[0]-point[0]
            ydis = center[1]-point[1]
            if(xdis != 0 and ydis != 0):
                tempAngle = math.atan(abs(xdis)/abs(ydis))*180/math.pi
                if(xdis > 0 and ydis > 0):
                    angle = -(tempAngle)  # 右旋
                if(xdis < 0 and ydis > 0):
                    angle = tempAngle
                if(xdis > 0 and ydis < 0):
                    angle = tempAngle-180
                if(xdis < 0 and ydis < 0):
                    angle = 180-tempAngle
            if(xdis == 0 and ydis < 0):
                angle = 180
            if(ydis == 0 and xdis > 0):
                angle = -90
            if(ydis == 0 and xdis < 0):
                angle = 90
            return angle

        def getCroppedBycenter(img, CharacterCenter, Rlength):
            '''
            根据图片中心点，和指定的正方形边长得到剪裁的图片
            输入：图像，多边形的中心点的坐标：list形式[x,y]，指定
            输出：剪裁好的图片
            '''
            CharacterCenter = [int(i) for i in CharacterCenter]
            span = int(Rlength/2)
            cropped = img[CharacterCenter[1]-span:CharacterCenter[1] +
                          span, CharacterCenter[0]-span:CharacterCenter[0]+span]
            return cropped

        def rotateImg(cropped, angle=0, pad=0, borderValue=0):
            '''
            得到旋转后图片
            输入：剪裁图片和旋转角度
            输出：将剪裁图片旋转angle的图像
            '''
            rows, cols = cropped.shape[:2]
            M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), angle, 1)
            dst = cv.warpAffine(cropped, M, (cols+pad, rows+pad),
                                borderValue=(borderValue, borderValue, borderValue))
            return dst
        angle = calcAngle(characterCenter, circleCenter)  # 图片的圆心坐标 旋转角度
        cropped = getCroppedBycenter(
            img, characterCenter, Rlength)  # 得到剪裁的字符图片
        result = rotateImg(cropped, angle, pad, boderValue)
        return result

    @staticmethod
    def get_raw_group(cluster_character_list, img, center, Rlength=200, cropRlength=100, boderValue=0):
        '''
        根据聚类好的信息 剪裁旋转字符图片
        输入：cluster_character_list（聚类好的字符信息）,boderValue=0在旋转后的边缘填充像素值，ratio=0.9 表示剪裁的大小依据 （半径差的ratio倍）
        输出：raw_group_list：表示裁剪并旋转后返回的聚类好的图片列表
        '''
        raw_group_list = []
        circleCenter = center
        Cstart = int((Rlength - cropRlength)/2)
        Cend = int(Cstart+cropRlength)
        for cluster in cluster_character_list:
            raw_group = []
            for character_info in cluster:
                characterCenter = character_info[0]
                result = Word_Classification.getPatternByCenter(
                    img, circleCenter, characterCenter, Rlength, boderValue=boderValue)  # self.img
                result = result[Cstart:Cend, Cstart:Cend]
                raw_group.append(result)
            raw_group_list.append(raw_group)
        return raw_group_list

    @staticmethod
    def get_redetect_info(pattern_list, result_list):
        """
        检测字符匹配的情况
        input: pattern_list:提供的正则表达式列表 result_list 网络检测的字符串列表
        return: message:对比之后返回的检测信息
        """
        def revise_re(re_list):
            """
            修改提供的正则表达式
            input: re_list 待修改的字符串列表
            return：re_revise_list 修改后的字符串列表
            """
            re_revise_list = []
            for pattern in re_list:
                re_revise = ""
                for char in pattern:
                    # if (char in ('0','D','O','Q')):
                    #     re_revise += "[0DOQ]"
                    # elif (char in ('7','T')):
                    #     re_revise += "[7T]"
                    # elif (char in ('S','5','2','Z')):
                    #     re_revise += "[S52Z]"
                    # elif (char in ('1','I','L')):
                    #     re_revise += "[1IL]"
                    # elif (char in ('B','6','8')):
                    #     re_revise += "[B68]"

                    # 适用于 2021-07-19\word\word_all.json 数据
                    if (char == 'B'):
                        re_revise += "[B8SQP9]"
                    elif (char == 'H'):
                        re_revise += "[HTP]"
                    elif (char == '0'):
                        re_revise += "[H08Q9]"
                    elif (char == '1'):
                        re_revise += "[19]"
                    elif (char == '6'):
                        re_revise += "[6F]"
                    elif (char == '2'):
                        re_revise += "[2P]"
                    elif (char == 'P'):
                        re_revise += "[PQ]"
                    elif (char == 'L'):
                        re_revise += "[LA5]"
                    elif (char == '8'):
                        re_revise += "[80]"
                    elif not(char.isalnum()):
                        re_revise += ".?"
                    else:
                        re_revise += char
                re_revise_list.append(re_revise)
            return re_revise_list

        pattern_list = revise_re(pattern_list)
        is_NG = False
        if(len(result_list) != len(pattern_list)):
            is_NG = True
            return is_NG

        for item in result_list:
            flag = False
            for pattern in pattern_list:
                match_obj = re.match(pattern, item)
                if (match_obj and len(item) == len(match_obj.group())):
                    flag = True
                    break
            if not flag:
                is_NG = True
                break
        return is_NG

    @staticmethod
    def get_center(circle_info):
        return np.mean(np.array(circle_info), axis=0)[:2].tolist()

    @classmethod
    def get_str_matchInfoUnit(cls, img, bbox_list, r_inner, r_outer, center, ratio=0.9, ratio_rwidth=1.7):
        '''
        通过图片相关信息，获取聚类且切分好的字符串
        输入：img：图片 bbox_list：所有字符的box xyxy的信息, r_inner：字符所在的区域的内圆半径,r_outer：字符所在的区域的外圆半径
        center：圆心 xy，pattern_list:提供的匹配字符串, ratio=0.9 表示剪裁的大小依据 （半径差的ratio倍） ratio_rwidth:判断字符group的基准 thresh：ratio_rwidth*r_width
        输出：str_bbox_list：表示裁剪并旋转后返回的聚类好的字符串列表 raw_group_list:切好旋转的原图
        '''
        if (len(bbox_list) == 0):
            return [], []
        img_position = cls.get_img_character_position(bbox_list, center)
        radius = r_inner+(r_outer-r_inner)/2
        r_width = r_outer-r_inner
        # set thresh 字符间弧长的间距不超过半径差的ratio_rwidth倍 则认为是相同group
        thresh = ratio_rwidth*r_width
        cluster_character_list = cls.cluster_character(
            img_position, radius, thresh=thresh)
        str_bbox_list = cls.find_strbbox(cluster_character_list)
        cropRlength = int(ratio*(r_outer-r_inner))
        Rlength = 2*cropRlength
        raw_group_list = cls.get_raw_group(
            cluster_character_list, img, center, Rlength, cropRlength)
        return str_bbox_list, raw_group_list

    def get_str_matchInfo(self, img, circles, pattern_list=[], ratio=0.9, ratio_rwidth=1.7):
        det_res = self.det_obj(img, circles)
        bbox_list = det_res['xyxys']
        center = self.get_center(circles)

        def find_distance(x):
            cx, cy = (x[0]+x[2])/2, (x[1]+x[3])/2
            return math.sqrt((cx-center[0])**2+(cy-center[1])**2)

        dis_list = list(map(find_distance, bbox_list))
        i_group_bbox_list = []
        o_group_bbox_list = []
        i_r_inner = circles[0][2]
        i_r_outer = circles[1][2]
        o_r_inner = circles[2][2]
        o_r_outer = circles[3][2]
        for i, dis in enumerate(dis_list):
            Info = bbox_list[i]
            bbox = [[Info[0], Info[1]], [Info[2], Info[3]]]
            if(dis >= i_r_inner and dis <= i_r_outer):
                i_group_bbox_list.append(bbox)
            if(dis >= o_r_inner and dis <= o_r_outer):
                o_group_bbox_list.append(bbox)
        i_str_bbox_list, i_raw_group_list = self.get_str_matchInfoUnit(
            img, i_group_bbox_list, i_r_inner, i_r_outer, center, ratio, ratio_rwidth)
        o_str_bbox_list, o_raw_group_list = self.get_str_matchInfoUnit(
            img, o_group_bbox_list, o_r_inner, o_r_outer, center, ratio, ratio_rwidth)
        raw_group_list = i_raw_group_list + o_raw_group_list
        str_bbox_list = i_str_bbox_list + o_str_bbox_list
        str_list = []
        for raw_group in raw_group_list:
            group_str = self.cc_obj.get_pred_str(raw_group)
            str_list.append(group_str)
        is_NG = Word_Classification.get_redetect_info(pattern_list, str_list)
        result = {"str_bbox_list": str_bbox_list,
                  "pattern_list": pattern_list, "str_list": str_list}
        return is_NG, result

    @staticmethod
    def get_detect_info(pattern_list, result_list):
        """
        检测字符匹配的情况（仅仅忽略非数字和字母的情况）
        input: pattern_list:提供的正则表达式列表(仅仅忽略非数字和字母的情况) result_list 网络检测的字符串列表
        return: message:对比之后返回的检测信息
        """
        def revise_re(re_list):
            """
            修改提供的正则表达式
            input: re_list 待修改的字符串列表
            return：re_revise_list 修改后的字符串列表
            """
            re_revise_list = []
            for pattern in re_list:
                re_revise = ""
                for char in pattern:
                    if not(char.isalnum()):
                        re_revise += "."
                    else:
                        re_revise += char
                re_revise_list.append(re_revise)
            return re_revise_list

        pattern_list = revise_re(pattern_list)
        is_NG = False
        if(len(result_list) != len(pattern_list)):
            is_NG = True
            return is_NG

        for item in result_list:
            flag = False
            for pattern in pattern_list:
                match_obj = re.match(pattern, item)
                if (match_obj and len(item) == len(match_obj.group())):
                    flag = True
                    break
            if not flag:
                is_NG = True
                break
        return is_NG

    def get_str_Info(self, img, bbox_list, r_inner, r_outer, center, pattern_list=[], ratio=0.9, ratio_rwidth=1.7):
        '''
        通过图片相关信息，获取聚类且切分好的字符串（仅仅忽略非数字和字母的情况）
        输入：img：图片 bbox_list：所有字符的box xyxy的信息, r_inner：字符所在的区域的内圆半径,r_outer：字符所在的区域的外圆半径
        center：圆心 xy，pattern_list:提供的匹配字符串, ratio=0.9 表示剪裁的大小依据 （半径差的ratio倍） ratio_rwidth:判断字符group的基准 thresh：ratio_rwidth*r_width
        输出：str_list：表示裁剪并旋转后返回的聚类好的字符串列表 message匹配结果信息（仅仅忽略非数字和字母的情况）
        '''
        img_position = self.get_img_character_position(bbox_list, center)
        radius = r_inner+(r_outer-r_inner)/2
        r_width = r_outer-r_inner
        # set thresh 字符间弧长的间距不超过半径差的ratio_rwidth倍 则认为是相同group
        thresh = ratio_rwidth*r_width
        cluster_character_list = self.cluster_character(
            img_position, radius, thresh=thresh)
        str_bbox_list = self.find_strbbox(cluster_character_list)
        cropRlength = int(ratio*(r_outer-r_inner))
        Rlength = 2*cropRlength
        raw_group_list = self.get_raw_group(
            cluster_character_list, img, center, Rlength, cropRlength)
        str_list = []
        for raw_group in raw_group_list:
            group_str = self.cc_obj.get_pred_str(raw_group)
            str_list.append(group_str)
        is_NG = Word_Classification.get_detect_info(pattern_list, str_list)
        result = {"str_bbox_list": str_bbox_list,
                  "pattern_list": pattern_list, "str_list": str_list}
        return is_NG, result
