import argparse
from models.albef.engine import ALBEF

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="./docs/demo.jpg", type=str)
    parser.add_argument("--query", default="boy with white hair", type=str)
    args = parser.parse_args()

    engine = ALBEF(model_id='ALBEF', device='cuda', templates='there is a {}')

    engine.visualize_groundvlp(image_path=args.image_path, query=args.query)



