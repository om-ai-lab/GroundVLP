import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_dir, ".."))

from utils.stanza_utils import find_agent_by_stanza
import json
from tqdm import tqdm

from Detic.predict import get_clip_embeddings
import torch
from torch.nn import functional as F


with open('./data/coco_80.txt', 'r') as f:
    word_list = f.readlines()
word_list = [word.strip() for word in word_list]
prompt = 'a photo of '
zs_weight = get_clip_embeddings(word_list, prompt=prompt)
zs_weight = F.normalize(zs_weight, p=2, dim=0)

def map_to_coco_label(pseudo_categories):
    res = []
    e_pseudo = get_clip_embeddings(pseudo_categories, prompt=prompt, permute=False)
    e_pseudo = F.normalize(e_pseudo, p=2, dim=1)

    similar = torch.mm(e_pseudo, zs_weight)
    similar = torch.nn.Softmax(dim=-1)(similar)

    top1 = torch.max(similar, dim=1)
    indices = top1.indices
    values = top1.values
    for i in range(len(pseudo_categories)):
        res.append(word_list[indices[i]])

    return res


if __name__ == '__main__':

    print("The prompt is {}".format(prompt))
    tasks = ['refcoco_val', 'refcoco_testA', 'refcoco_testB', 'refcoco+_val', 'refcoco+_testA', 'refcoco+_testB', 'refcocog_val','refcocog_test']
    pc_path = './data/pc'
    os.makedirs(pc_path, exist_ok=True)
    
    for task in tasks[:]:
       
        m = json.load(open(f'./data/{task}_info.json'))
        json_datas = []
        for x in tqdm(m, desc="Progress"):
            pseudo_categories = []
            for sent in x['sentences']:
                agent = find_agent_by_stanza(sent)
                pseudo_categories.append(agent)
            mapped = map_to_coco_label(pseudo_categories)
            # mapped = [mc if pc!="[UNK]" else pc for mc, pc in zip(mapped, pseudo_categories)]
            x['pseudo_categories'] = mapped
            json_datas.append(x)
        with open(os.path.join(pc_path, f'{task}_info.json'), 'w') as f:
            json.dump(json_datas, f)
