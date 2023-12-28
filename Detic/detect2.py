"""
Author:Shen Haozhan @ ZJU
Date:2022--11--22
"""
import json_files

import torch

from predict import Predictor, get_clip_embeddings

predictor = Predictor()
predictor.setup()
emb_person = get_clip_embeddings(['person'], permute=False)


def get_bbox_by_detic(path, category):
    emb_category = get_clip_embeddings([category], permute=False)
    similarity_with_person = torch.cosine_similarity(emb_person, emb_category)
    print(similarity_with_person)
    if similarity_with_person >= 0.9:
        category = 'person'
    instances, _ = predictor.predict(path, 'custom', category)
    return instances.pred_boxes.tensor.tolist()


if __name__ == '__main__':
    # paths = ['./images/COCO_train2014_000000000077.jpg',
    #          './images/COCO_train2014_000000000154.jpg',
    #          './images/COCO_train2014_000000000165.jpg',
    #          './images/2359985.jpg']
    # categorys = ['person,shirt']
    # for path, category in zip(paths, categorys):
    #     boxes = predictor.predict(path, 'custom', category)[0].pred_boxes.tensor.tolist()
    #     print(boxes)
    image_folder = "/data9/shz/dataset/coco/train2014"
    path = 'COCO_train2014_000000581282.jpg'
    import os
    path = os.path.join(image_folder, path)
    category = 'custom'
    instance = predictor.predict(path, 'custom', 'person', 0.5)
    print(instance)
    box_list = instance.pred_boxes.tensor.tolist()
    from draw_box import draw_boxes
    draw_boxes(image_path=path,box_list=box_list, word_list=['person']*len(box_list),output_path='/data/shz/project/groundvlp/GroundVLP/Detic/output/output.jpg')
    # categorys = ['woman', 'baby', 'oxne',
    #              'boy', 'girl', 'man', 'human', 'player']
    # emb_categorys = get_clip_embeddings(categorys, permute=False)
    # s = torch.cosine_similarity(emb_person, emb_categorys)
    # for i, category in enumerate(categorys):
    #     # emb_category = get_clip_embeddings([category], permute=False)
    #     # similarity_with_person = torch.cosine_similarity(emb_person, emb_category)
    #     if s[i] > 0.89:
    #         categorys[i] = 'person'
    # print(categorys)