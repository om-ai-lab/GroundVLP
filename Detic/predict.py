import json
import sys
import cv2
import tempfile
from pathlib import Path
import cog
import time
import torch
import pickle

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os

cur_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_dir, "third_party/CenterNet2"))
sys.path.insert(0, cur_dir)
# Detic libraries
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder


class Predictor(cog.BasePredictor):
    def setup(self):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(
            "Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        cfg.MODEL.WEIGHTS = os.path.join(cur_dir, '../checkpoints', 'Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.MODEL.MASK_ON = False
        cfg.MODEL.DEVICE = 'cuda'
        self.predictor = DefaultPredictor(cfg)
        self.BUILDIN_CLASSIFIER = {
            'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
            'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
            'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
            'coco': 'datasets/metadata/coco_clip_a+cname.npy',
        }
        self.BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'objects365': 'objects365_v2_val',
            'openimages': 'oid_val_expanded',
            'coco': 'coco_2017_val',
        }

    def predict(self, image, vocabulary, custom_vocabulary, threshold_set):
        image = cv2.imread(str(image))
        if not vocabulary == 'custom':
            metadata = MetadataCatalog.get(self.BUILDIN_METADATA_PATH[vocabulary])
            classifier = self.BUILDIN_CLASSIFIER[vocabulary]
            num_classes = len(metadata.thing_classes)
            reset_cls_test(self.predictor.model, classifier, num_classes)

        else:
            assert custom_vocabulary is not None and len(custom_vocabulary.split(',')) > 0, \
                "Please provide your own vocabularies when vocabulary is set to 'custom'."
            metadata = MetadataCatalog.get(str(time.time()))
            metadata.thing_classes = custom_vocabulary.split(',')
            classifier = get_clip_embeddings(metadata.thing_classes)
            num_classes = len(metadata.thing_classes)
            reset_cls_test(self.predictor.model, classifier, num_classes)
            # Reset visualization threshold
            output_score_threshold = threshold_set
            for cascade_stages in range(len(self.predictor.model.roi_heads.box_predictor)):
                self.predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold

        outputs = self.predictor(image)
        return outputs["instances"]


text_encoder = build_text_encoder(pretrain=True)
text_encoder.eval()


def get_clip_embeddings(vocabulary, prompt='a ', permute=True):
    texts = [prompt + x for x in vocabulary]
    with torch.no_grad():
        if permute:
            emb = text_encoder(texts).detach().permute(1, 0).contiguous()
        else:
            emb = text_encoder(texts).detach()
    return emb

predictor = Predictor()
predictor.setup()
emb_person = get_clip_embeddings(['person'], permute=False)

detic_boxes = {}

with open('./Detic/json_files/detic_boxes_rec_scores_015.json') as f:
    detic_boxes = json.load(f)
    print('detic_boxes has been loaded...')

with open('./Detic/json_files/id_to_proposals_rec.json') as f:
    all_proposals_refcoco = json.load(f)
    print('json of all rec proposals has been loaded...')

def get_bbox_by_detic_rec(path, category, threshold_set, unk=False):
    image_name = path.split('/')[-1]
    name = image_name.replace(".jpg", "").split("_")[-1]
    image_id = str(int(name))

    if unk:
        boxes_category = all_proposals_refcoco[image_id]
        od_scores = None
        return boxes_category, od_scores

    instances = predictor.predict(path, 'custom', category, threshold_set)
    boxes_category = instances.pred_boxes.tensor.tolist()
    od_scores = instances.scores.tolist()

    assert len(boxes_category) == len(od_scores)
    if len(boxes_category) == 0:
        boxes_category = all_proposals_refcoco[image_id]
        od_scores = None
    return boxes_category, od_scores

def get_bbox_by_detic_rec_general(path, category, threshold_set):
    boxes_category = []
    od_scores = []
    while len(boxes_category) == 0:
        instances = predictor.predict(path, 'custom', category, threshold_set)
        boxes_category = instances.pred_boxes.tensor.tolist()
        od_scores = instances.scores.tolist()
        threshold_set /= 2
    return boxes_category, od_scores

# def get_bbox_by_detic_rec(path, category, threshold_set, unk=False):
#     image_name = path.split('/')[-1]
#     name = image_name.replace(".jpg", "").split("_")[-1]
#     image_id = str(int(name))

#     if unk:
#         boxes_category = all_proposals_refcoco[image_id]
#         od_scores = None
#         return boxes_category, od_scores
        
#     if category in detic_boxes[image_id].keys():
#         boxes_category, od_scores = detic_boxes[image_id][category]
#         if len(boxes_category) > 0:
#             while od_scores[-1] < threshold_set:
#                 od_scores.pop()
#                 boxes_category.pop()
#                 if len(boxes_category)==0:
#                     break
#     else:
#         instances = predictor.predict(path, 'custom', category, threshold_set)
#         boxes_category = instances.pred_boxes.tensor.tolist()
#         od_scores = instances.scores.tolist()
#     assert len(boxes_category)==len(od_scores)
#     if len(boxes_category)==0:
#         boxes_category = all_proposals_refcoco[image_id]
#         od_scores = None
#     return boxes_category, od_scores

#
# def save_detic_boxes():
#     with open(f'/content/drive/MyDrive/VL-CheckList/pkls/detic_boxes.pkl', 'wb') as f:
#         pickle.dump(detic_boxes, f)
#     print(f'pkl file of detic_boxes has been saved...')
