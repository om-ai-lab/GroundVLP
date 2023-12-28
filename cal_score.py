import argparse
import json
import os
from tqdm import tqdm
from models.albef.engine import ALBEF

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", default="/data9/shz/dataset/coco/train2014", type=str)
    parser.add_argument("--eval_data", default="refcoco+_testA", type=str)
    parser.add_argument("--model_id", default="ALBEF", type=str)
    parser.add_argument("--use_gt_category", action='store_true', default=True)
    args = parser.parse_args()

    eval_datas = args.eval_data.split(',')
    for task in eval_datas:
        m = json.load(open(f"./data/{task}_info_pc.json"))
        data = []
        engine = ALBEF(model_id=args.model_id)

        for x in m:
            img_name = x['file_name']
            img_file = os.path.join(args.image_folder, img_name)
        
            if not os.path.exists(img_file):
                print(f"{img_name} are not loaded")
            else:
                data.append((img_file, x['sentences'], x['gt_bbox'], x['category'], x['pseudo_categories']))
        
        print("Finish Loading...")

        block_num = 8
        total = 0
        total_right = 0
        for d in tqdm(data, desc="Progress"):
            img_path = d[0]
            texts = d[1]       
            gt_bbox = [d[2][0], d[2][1], d[2][0] + d[2][2], d[2][1] + d[2][3]]
            category = d[3]
            pseudo_categories = d[4]
            # image_id = d[4]
            right_pred = engine.get_results_for_rec(
                image_path=img_path,
                texts=texts,
                gt_bbox=gt_bbox,
                block_num=block_num,
                category=category,
                mapped_categories=None if args.use_gt_category else pseudo_categories
            )
            
            total += len(texts)
            total_right += right_pred
        print(task)
        acc = 100*total_right/total
        print(f"acc:{round(acc, 2)}%")
        print('='*50)
       

