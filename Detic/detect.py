"""
Author:Shen Haozhan @ ZJU
Date:2022--11--15
"""
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss
from pprint import pprint

from PIL import Image

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo

# constants
WINDOW_NAME = "Detic"


def setup_cfg(args):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file('./configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml')
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later

    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    # ------------------------------------
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        nargs=argparse.REMAINDER,
    )
    return parser


mp.set_start_method("spawn", force=True)
args = get_parser().parse_args()
# args.config_file = "./configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
args.config_file = "./configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
args.opts = ['MODEL.WEIGHTS', './models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth']
args.vocabulary = 'custom'
args.custom_vocabulary = 'person'
image_id = '101958970'
args.confidence_threshold = 0.3
cfg = setup_cfg(args)
demo = VisualizationDemo(cfg, args)


def get_bbox_by_detic(path, category):
    img = read_image(path, format="BGR")
    predictions, labels = demo.run_on_image(img)
    classes = predictions['instances'].get('pred_classes').tolist()
    boxes = predictions['instances'].get('pred_boxes').tensor.tolist()
    print(labels)
    res = []
    class_names = [labels[i] for i in classes]
    for name, box in zip(class_names, boxes):
        if name == category:
            res.append(box)

    return res


if __name__ == "__main__":
    # setup_logger(name="fvcore")
    # logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    # boxes = get_bbox_by_detic('desk.jpg', 'book')
    # print(boxes)

    data_root = '/Users/shen/PycharmProjects/flicker30k/'
    image_path = os.path.join(data_root, f'flickr30k-images/{image_id}.jpg')

    args.input = '/Users/shen/PycharmProjects/flicker30k/flickr30k-images/101958970.jpg'
    args.output = './output/out.jpg'

    path = image_path

    img = read_image(path, format="BGR")
    start_time = time.time()
    predictions, visualized_output, labels = demo.run_on_image(img)
    # print(Image.open(image_path).size)
    print(predictions["instances"].pred_boxes.tensor.tolist())



    # print(len(labels))
    # with open('./datasets/oid.txt', 'w') as f:
    #     for label in labels:
    #         f.write(label)
    #         f.write('\r\n')
    # print(predictions['instances'].get('pred_boxes').tensor.tolist())
    # logger.info(
    #     "{}: {} in {:.2f}s".format(
    #         path,
    #         "detected {} instances".format(len(predictions["instances"]))
    #         if "instances" in predictions
    #         else "finished",
    #         time.time() - start_time,
    #     )
    # )

    out_filename = args.output
    visualized_output.save(out_filename)

    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    # if cv2.waitKey(0) == 27:
    #     break  # esc to quit
