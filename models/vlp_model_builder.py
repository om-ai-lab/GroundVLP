from typing import Any
from .vlp_model import VLPModel
from models.albef.engine import ALBEF

class VLPModelBuilder:
    MODEL_ID_MAP = {
        'ALBEF': ALBEF,
        'TCL': ALBEF,
    }
    def __call__(self, model_id, **kwargs) -> VLPModel:
        if model_id not in self.MODEL_ID_MAP.keys():
            raise ValueError('Meaningless model_id: {}, you have to choose from: {}'.format(model_id, self.MODEL_ID_MAP.keys()))
        return self.MODEL_ID_MAP[model_id](model_id, **kwargs)