import os
from typing import List
import torch
import re

from models.vlp_model import VLPModel
from .VL_Transformer_ITM import VL_Transformer_ITM, transform, getAttMap
from models.albef.models.tokenization_bert import BertTokenizer
import torch.nn.functional as F

from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np
from Detic.draw_box import draw_boxes_cv2

class ALBEF(VLPModel):
    def __init__(self, model_id, device='cuda', templates = 'there is a {}'):
        super().__init__(model_id, device, templates=templates)
    
    def load_model(self, model_id):
        if model_id is None:
            raise Exception("Model ID cannot be None.")

        if not self._models.has(model_id):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = VL_Transformer_ITM(text_encoder='bert-base-uncased',
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
        texts_prompt = [self.templates.format(text) for text in texts]

        text_input = tokenizer(texts_prompt, return_tensors="pt", padding='max_length',
                               truncation=True, max_length=50).to(self.device)
        
        output = model(image, text_input)
        loss = output[:, 1].sum()
        model.zero_grad()
        loss.backward()

        num_patch = int(model.visual_encoder.patch_embed.num_patches**0.5)
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
        # 1 means the removal of [CLS] 
        start_idx = len(tokenizer(self.templates.format(""), return_tensors='pt').input_ids[0]) - 1

        for z, text in enumerate(texts):
            # Use mapped_category
            if mapped_categories:
                 boxes_category, od_scores = self.get_bbox_for_rec(image_path, mapped_categories[z], threshold=0.3, unk=(mapped_categories[z]=='[UNK]')) 
            main_words_ids, id_map = self.find_main_words(text, start_idx, tokenizer)
            all_focus_ids = []

            for word_id in main_words_ids:
                all_focus_ids.extend(id_map[word_id])
            
            find_words = (len(all_focus_ids)>0)
            # Add [CLS]
            all_focus_ids.append(0)

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
        od_scores,
        use_weighted_grade=True
    ):
        # cal box_output
        max_score = 0
        box_output = [0, 0, 0, 0]
        if not use_weighted_grade:
            od_scores = None
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
        if gt_bbox and self.cal_iou(box_output, gt_bbox) >= 0.5:
            x = 1
        return x, box_output
    
    def visualize_groundvlp(
        self,
        image_path,
        query,
        block_num=8
    ):
        model, tokenizer = self.load_model(self.model_id)

        image_pil = Image.open(image_path).convert('RGB')
        image = transform(image_pil).unsqueeze(0).to(self.device)

        model.text_encoder.encoder.layer[block_num].crossattention.self.save_attention = True
        
        text_input = tokenizer(self.templates.format(query), return_tensors="pt").to(self.device)

        output = model(image, text_input)
        loss = output[:, 1].sum()
        model.zero_grad()
        loss.backward()

        num_patch = int(model.visual_encoder.patch_embed.num_patches**0.5)

        with torch.no_grad():
            mask = text_input.attention_mask.view(text_input.attention_mask.size(0), 1, -1, 1, 1).cpu()

            cams = model.text_encoder.encoder.layer[block_num].crossattention.self.get_attention_map().cpu()
            grads = model.text_encoder.encoder.layer[block_num].crossattention.self.get_attn_gradients().cpu()

            cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, num_patch, num_patch) * mask
            grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, num_patch, num_patch) * mask

            gradcams = cams * grads
        
        # Find agent
        agent = self.find_agent(query)
        mapped_coco_label = self.map_to_coco_label(agent)
        # Obtain the boxes_category
        boxes_category, od_scores = self.get_bbox_for_rec(image_path, mapped_coco_label, threshold=0.15, general=True)

        # Visual-Word Attention Aggregation
        start_idx = len(tokenizer(self.templates.format(""), return_tensors='pt').input_ids[0]) - 1
        main_words_ids, id_map = self.find_main_words(query, start_idx, tokenizer)
        all_focus_ids = []

        for word_id in main_words_ids:
            all_focus_ids.extend(id_map[word_id])
        
        find_words = (len(all_focus_ids)>0)
        # Add [CLS]
        all_focus_ids.append(0)

        num_effective_text_token = text_input.attention_mask[0].count_nonzero().item()
        gradcam = gradcams[0]
        gradcam = gradcam.mean(0)[all_focus_ids, ...].mean(0) if find_words else \
            gradcam.mean(0)[:num_effective_text_token, ...].mean(0)
        gradcam = gradcam.view(1, 1, num_patch, num_patch)
        gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
                                mode='bicubic', align_corners=False).squeeze()
        
        self.show_groundvlp(
            image_path=image_path, 
            query=query, gradcam=gradcam, 
            category=mapped_coco_label, 
            boxes_category=boxes_category, 
            od_scores=od_scores
        )
        
        
    
    def show_groundvlp(
        self,
        image_path,
        query,
        gradcam,
        category,
        boxes_category,
        od_scores
    ):
        num_image = 4
        fig, ax = plt.subplots(num_image, 1, figsize=(15, 5 * num_image))

        bgr_image = cv2.imread(image_path)
        ax[0].imshow(bgr_image[:, :, ::-1])
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[0].set_xlabel(query, fontsize=15)

        rgb_image = cv2.imread(image_path)[:, :, ::-1]
        rgb_image = np.float32(rgb_image) / 255

        gradcam_image = getAttMap(rgb_image, gradcam)
        np.clip(gradcam_image, 0., 1., out=gradcam_image)
        # print(np.max(gradcam_image))
        ax[1].imshow(gradcam_image)
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_xlabel(query, fontsize=20)

        # visualize detic bbox
        cv2_img_2 = draw_boxes_cv2(image_path=image_path, box_list=boxes_category)
        ax[2].imshow(cv2_img_2[:, :, ::-1])
        ax[2].set_yticks([])
        ax[2].set_xticks([])
        ax[2].set_xlabel(f'category: {category}', fontsize=20)

        _, box_output = self.cal_score(gradcam=gradcam, gt_bbox=None, boxes_category=boxes_category, od_scores=od_scores, use_weighted_grade=True)

        cv2_img_3 = draw_boxes_cv2(image_path=image_path, box_list=[box_output])
        ax[3].imshow(cv2_img_3[:, :, ::-1])
        ax[3].set_yticks([])
        ax[3].set_xticks([])
        ax[3].set_xlabel('box_output', fontsize=20)

        save_dir = "./output"
        os.makedirs(save_dir, exist_ok=True)
        image_name = image_path.split('/')[-1]
        save_path = os.path.join(save_dir, image_name)
        plt.savefig(save_path)
        plt.close('all')


