"""
Author:Shen Haozhan @ ZJU
Date:2023--01--11
"""
import json
import os
from pprint import pprint

import cv2
import numpy as np
import random

import torch

from detectron2.data.detection_utils import read_image
from flickr30k_entities_utils import get_annotations, get_sentence_data
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures.instances import Instances
from detectron2.data.catalog import Metadata, MetadataCatalog
from detectron2.structures.boxes import Boxes

mp_color = {}


def random_bgr(cls):
    if cls in mp_color.keys():
        return mp_color[cls]
    else:
        b = random.randint(0, 255)
        random.seed(random.randint(1,100))
        g = random.randint(0, 255)
        random.seed(random.randint(101, 200))
        r = random.randint(0, 255)
        mp_color[cls] = (b, g, r)
        return b, g, r


# def draw_boxes(image_path: str, box_list: list, word_list: list):
#     assert len(box_list)==len(word_list)
#
#     im = cv2.imread(image_path)
#     # blk = np.zeros(im.shape, np.uint8)
#     # white_blk = np.ones(im.shape, np.uint8) * 255
#
#     for box, word in zip(box_list, word_list):
#         x1, y1, x2, y2 = box
#         # b, g, r = random_bgr(word)
#         # t_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_TRIPLEX,0.6, 1)[0]
#         # ptLeftTop = np.array([x1+2, y1+2])
#         # textlbottom = ptLeftTop + np.array(list(t_size))
#
#         im = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 100, 10), 3)
#         # blk = cv2.rectangle(blk, tuple(ptLeftTop), tuple(textlbottom), (255, 255, 255), -1)
#         # ptLeftTop[1] = ptLeftTop[1] + (t_size[1] / 2 + 4)
#         # blk = cv2.putText(blk, word, tuple(ptLeftTop), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
#
#     # im = cv2.addWeighted(im, 1.0, blk, 1, 1)
#     # im = cv2.addWeighted(im, 1.0, white_blk, 0.1, 1)
#
#     cv2.imwrite('./output/test/box_image.jpg', im)

def draw_boxes(image_path: str,
               box_list: list,
               word_list: list,
               output_path:str,
               k = 0):
    metadata = MetadataCatalog.get(f"__unused_{k}")
    metadata.thing_classes = word_list
    image = read_image(image_path, format="BGR")[:, :, ::-1]
    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)

    instance = Instances((image.shape[0], image.shape[1]))
    instance.pred_boxes = Boxes(box_list)
    instance.pred_classes = torch.tensor(range(len(word_list)), dtype=torch.int)

    vis_output = visualizer.draw_instance_predictions(predictions=instance)
    vis_output.save(output_path)

if __name__ == '__main__':
    # m = json.load(open('/Users/shen/PycharmProjects/OPTtest/json/region_descriptions.json'))
    # image_id = 1592530
    # image_path = f'/Users/shen/PycharmProjects/OPTtest/image/{image_id}.jpg'
    image_path = ''
    box_list = [[0,0,0,0]]
    word_list = ['']

    output_path = ''

    draw_boxes(image_path=image_path,
               box_list=box_list,
               word_list=word_list,
               output_path=output_path)



# if __name__ == '__main__':
#     image_id = '2448393373'
#     data_root = '/Users/shen/PycharmProjects/flicker30k/'
#     image_path = os.path.join(data_root, f'flickr30k-images/{image_id}.jpg')
#     # image_path = '/Users/shen/PycharmProjects/Detic/Detic/output/cam.png'
#     coref_datas = get_sentence_data(os.path.join(data_root,
#                                 f'flickr30k_entities-master/Sentences/{image_id}.txt'))
#     annotation_data = get_annotations(os.path.join(data_root,
#                                 f'flickr30k_entities-master/Annotations/{image_id}.xml'))
#     # output_path = f'/Users/shen/PycharmProjects/Detic/Detic/output/flicker_test/{image_id}_gt.jpg'
#     # box_list = [[69, 28, 400, 374], [244, 197, 353, 369], [325, 0, 361, 43]]
#     # word_list = ['1', '1', '1']
#     # draw_boxes(image_path, box_list, word_list)
#     for idx in range(5):
#         print(idx)
#         print(coref_datas[idx]['sentence'])
#
#         box_list = []
#         cls_list = []
#         word_list = []
#
#         for phrase in coref_datas[idx]['phrases']:
#             if phrase['phrase_id'] in annotation_data['nobox']:
#                 continue
#             if phrase['phrase_id'] not in annotation_data['boxes'].keys():
#                 continue
#             boxes = annotation_data['boxes'][phrase['phrase_id']]
#             box_list.extend(boxes)
#             cls_list.extend([phrase['phrase_id']]*len(boxes))
#             word_list.extend([phrase['phrase']]*len(boxes))
#             print(phrase['phrase'],'-->',phrase['phrase_type'])
#
#         output_path = f'/Users/shen/PycharmProjects/Detic/Detic/output/flicker_test/{image_id}_{idx + 1}_gt.jpg'
#         draw_boxes(image_path, box_list, word_list, output_path, k=idx)

