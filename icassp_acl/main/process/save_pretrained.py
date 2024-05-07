import torch
import argparse
from transformers import AutoTokenizer, T5Config
from multidoc2dial.unigdd import UniGdd

args = argparse.ArgumentParser()
args.add_argument('name', type=str)
args = args.parse_args()
pretrained_path = '/mnt/data/huairang/Doc2dial/unigdd/pretrained/t5-3b'
tokenizer = AutoTokenizer.from_pretrained('t5-3b')
config = T5Config.from_pretrained(pretrained_path)
model = UniGdd.from_pretrained(pretrained_path, config=config)
checkpoint = torch.load(f'/mnt/data/huairang/Doc2dial/unigdd/checkpoint/{args.name}/model_best.pth.tar',
                        map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

target_path = f'/mnt/data/fucheng/doc2dial/hm_{args.name}'
tokenizer.save_pretrained(target_path)
model.save_pretrained(target_path)
