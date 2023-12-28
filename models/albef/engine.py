import os
from typing import List
import torch
import re

from models.vlp_model import VLPModel
from .VL_Transformer_ITM import VL_Transformer_ITM, transform
from transformers import BertTokenizer
import torch.nn.functional as F

from PIL import Image
from utils.stanza_utils import find_main_words
from utils.cal_utils import cal_iou


class ALBEF(VLPModel):
    def __init__(self, model_id, device='cuda', templates = 'there is a {}'):
        super().__init__(model_id, device, templates=templates)
    
    def load_model(self, model_id):
        if model_id is None:
            raise Exception("Model ID cannot be None.")

        if not self._models.has(model_id):
            tokenizer = BertTokenizer.from_pretrained('/data9/shz/ckpt/bert-base-uncased')
            model = VL_Transformer_ITM(text_encoder='/data9/shz/ckpt/bert-base-uncased',
                                       config_bert=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                'config_bert.json'))
            checkpoint_path = os.path.join(self.checkpoint_dir, f'{model_id}.pth')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            checkpoint = {k.replace('.bert', ''): v for k, v in checkpoint.items()}

            msg = model.load_state_dict(checkpoint, strict=False)
            model.eval()
            model.to(self.device)
            self._models.put(model_id, (model, tokenizer))

        return self._models.get(model_id)

    def pre_caption(self, caption, max_words=50):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])
        return caption
    # 68.32%
    def get_results_for_rec(
        self, 
        image_path: str, 
        texts: List[str], 
        gt_bbox: List[int], 
        block_num: int = 8, 
        category='', 
        mapped_categories=None
    ):
        model, tokenizer = self.load_model(self.model_id)

        image_pil = Image.open(image_path).convert('RGB')
        image = transform(image_pil).unsqueeze(0)
        image = torch.cat([image] * len(texts), dim=0).to(self.device)

        model.text_encoder.encoder.layer[block_num].crossattention.self.save_attention = True
        texts_prompt = [self.templates.format(self.pre_caption(text)) for text in texts]

        text_input = tokenizer(texts_prompt, return_tensors="pt", padding='max_length',
                               truncation=True, max_length=50).to(self.device)
        
        output = model(image, text_input)
        loss = output[:, 1].sum()
        model.zero_grad()
        loss.backward()

        num_patch = model.visual_encoder.image_size // model.visual_encoder.patch_size
        right_pred = 0

        with torch.no_grad():
            mask = text_input.attention_mask.view(text_input.attention_mask.size(0), 1, -1, 1, 1).cpu()

            cams = model.text_encoder.encoder.layer[block_num].crossattention.self.get_attention_map().cpu()
            grads = model.text_encoder.encoder.layer[block_num].crossattention.self.get_attn_gradients().cpu()

            cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, num_patch, num_patch) * mask
            grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, num_patch, num_patch) * mask

            gradcams = cams * grads
        
        # Use gt category
        if not mapped_categories:
            boxes_category, od_scores = self.get_bbox_for_rec(image_path, category)
        # 2 means the removal of [CLS] and [SEP]
        start_idx = len(tokenizer(self.templates.format(""), return_tensors='pt').input_ids[0]) - 2
        # print("start_idx:", start_idx)
        # print(tokenizer.convert_ids_to_tokens(tokenizer(self.templates.format(""), return_tensors='pt').input_ids[0]))
        for z, text in enumerate(texts):
            # Use mapped_category
            if mapped_categories:
                 boxes_category, od_scores = self.get_bbox_for_rec(image_path, mapped_categories[z])
            # main_words_ids, id_map = find_main_words(text, start_idx, tokenizer)
            all_focus_ids = []

            # for word_id in main_words_ids:
            #     all_focus_ids.extend(id_map[word_id])
            
            find_words = (len(all_focus_ids)>0)
            # Add [CLS]
            all_focus_ids.append(0)
            # print(111, z )
            # print(tokenizer.convert_ids_to_tokens(text_input.input_ids[z]))
            # print(tokenizer.convert_ids_to_tokens(text_input.input_ids[z][all_focus_ids]))

            num_effective_text_token = text_input.attention_mask[z].count_nonzero().item()
            gradcam = gradcams[z]
            gradcam = gradcam.mean(0)[all_focus_ids, ...].mean(0) if find_words else \
                gradcam.mean(0)[:num_effective_text_token, ...].mean(0)
            gradcam = gradcam.view(1, 1, num_patch, num_patch)
            gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
                                    mode='bicubic', align_corners=False).squeeze()
            x, _ = self.cal_score(gradcam, gt_bbox, boxes_category, od_scores)
            right_pred += x

        return right_pred
            
    
    def cal_score(
        self,
        gradcam,
        gt_bbox,
        boxes_category,
        od_scores
    ):
        # cal box_output
        max_score = 0
        box_output = [0, 0, 0, 0]
        for i, det in enumerate(boxes_category):
            score = gradcam[int(det[1]):int(det[3]), int(det[0]):int(det[2])]
            area = (det[3] - det[1]) * (det[2] - det[0])
            score = score.sum()
            score /= area ** 0.5

            if od_scores is not None:
                s = od_scores[i]
                coefficient = s
                score *= coefficient

            if score > max_score:
                max_score = score
                box_output = det[:4]
        x = 0
        if cal_iou(box_output, gt_bbox) >= 0.5:
            x = 1
        return x, box_output