"""
Author:Shen Haozhan @ ZJU
Date:2023--05--24
"""
import os

import tqdm

from predict import Predictor
import json
import torch

def cal_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    #Intersection area
    # Intersection area 这里没有对inter_area为负的情况进行判断，后面计算出来的IOU就可能是负的
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


if __name__ == '__main__':
    predictor = Predictor()
    predictor.setup()
    m = json.load(open('./json_files/res_jsons/refcoco_val_info_pc.json'))
    data_root = '/Users/shen/PycharmProjects/refcoco/refer/data/llm_json/train2014'
    num_acc = 0
    total = 0
    for x in tqdm.tqdm(m, ncols=100, total=len(m)):
        category = x['category']
        path = os.path.join(data_root, x['file_name'])
        candidate_boxes = predictor.predict(path, 'custom', category, 0.15).pred_boxes.tensor
        for query in x['sentences']:
            total += 1
            instances = predictor.predict(path, 'custom', query, 0.01)
            box_list = instances.pred_boxes.tensor.tolist()
            box_list = box_list if len(box_list) > 0 else [[0,0,0,0]]
            flag = False
            ans_box = None
            for box in box_list:
                res = bbox_iou(candidate_boxes, torch.tensor([box]))
                val, idx = torch.max(res, 0)
                if val > 0.9:
                    ans_box = candidate_boxes[idx].tolist()
                    break
            if ans_box == None:
                ans_box = [0,0,0,0]
            if cal_iou(ans_box, x['gt_bbox']) > 0.5:
                num_acc += 1
    acc = 100 * num_acc / total
    print(f"acc:{round(acc, 2)}%")



    # path = '/Users/shen/PycharmProjects/Detic/Detic/images/COCO_train2014_000000571563.jpg'
    # query = 'person in gray and white plaid shirt'
    #
    # instances = predictor.predict(path, 'custom', query, 0.01)
    #
    # print(instances)
    # print(instances.pred_boxes.tensor.tolist())