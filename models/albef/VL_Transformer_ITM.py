from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters

from albef.models.tokenization_bert import BertTokenizer
from albef.models.vit import VisionTransformer
from albef.models.xbert import BertConfig, BertModel
from skimage import transform as skimage_transform

import torch
from torch import nn
from torchvision import transforms
import re
from PIL import Image

class VL_Transformer_ITM(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config_bert=''
                 ):
        super().__init__()

        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = VisionTransformer(
            img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        self.itm_head = nn.Linear(768, 2)

    def forward(self, image, text):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        output = self.text_encoder(text.input_ids.to(image.device),
                                   attention_mask=text.attention_mask.to(image.device),
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts.to(image.device),
                                   return_dict=True,
                                   )
        vl_embeddings = output.last_hidden_state[:, 0, :]
        # print(vl_embeddings)
        vl_output = self.itm_head(vl_embeddings)
        # _,pred = torch.max(vl_output,1)
        # print(type(vl_output))
        return vl_output


def getAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])




