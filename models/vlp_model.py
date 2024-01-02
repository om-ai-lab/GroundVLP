from typing import List

from Detic.predict import get_bbox_by_detic_rec, get_bbox_by_detic_rec_general
from .helpers import LRUCache
from utils.cal_utils import cal_iou
from utils.stanza_utils import find_main_words, find_agent_by_stanza
from utils.map_to_coco_label import map_to_coco_label

class VLPModel:
    MAX_CACHE = 20
    def __init__(self, model_id, device='cuda', templates = 'there is a {}',
                 checkpoint_dir = './checkpoints'):
        self._models = LRUCache(self.MAX_CACHE)
        self.model_id = model_id
        self.device = device
        self.templates = templates
        self.checkpoint_dir = checkpoint_dir
    
    def get_bbox_for_rec(self, image_path, category, threshold=0.15, unk=False, general=False):
        if general:
            return get_bbox_by_detic_rec_general(image_path, category, threshold)
        else:
            return get_bbox_by_detic_rec(image_path, category, threshold, unk=unk)
    
    def cal_iou(self, box1, box2):
        return cal_iou(box1, box2)

    def find_main_words(self, sent, start_idx, tokenizer):
        return find_main_words(sent, start_idx, tokenizer)
    
    def find_agent(self, sent):
        return find_agent_by_stanza(sent)
    
    def map_to_coco_label(self, agent):
        return map_to_coco_label([agent])[0]
    
    def get_results_for_rec(
        self, 
        image_path: str, 
        texts: List[str], 
        gt_bbox: List[int], 
        block_num: int = 8, 
        category='', 
        mapped_categories=None
    ):
        pass

    def cal_score(
        self,
        gradcam,
        gt_bbox,
        boxes_category,
        od_scores
    ):
        pass 

    
