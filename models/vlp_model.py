
from Detic.predict import get_bbox_by_detic_rec
from helpers import LRUCache

class VLPModel:
    MAX_CACHE = 20
    def __init__(self, model_id, device='cuda', templates = 'there is a {}',
                 checkpoint_dir = './checkpoints'):
        self._models = LRUCache(self.MAX_CACHE)
        self.model_id = model_id
        self.device = device
        self.templates = templates
        self.checkpoint_dir = checkpoint_dir
    
    def get_bbox_for_rec(image_path, category, threshold=0.15):
        return get_bbox_by_detic_rec(image_path, category, threshold)