import json
import argparse

args = argparse.ArgumentParser()
args.add_argument('name', type=str)
args.add_argument('--code', type=str, default='unigdd')
args = args.parse_args()

result_utterance = []
result_grounding = []
with open(f'/mnt/data/huairang/Doc2dial/{args.code}/checkpoint/{args.name}/result-test.json', 'r') as f:
    samples = json.load(f)
    for id_, hypothesis in zip(samples['id_list'], samples['outputs']):
        grounding = hypothesis.split('<grounding> ')[-1].split('<response>')[0].strip()
        utterance = hypothesis.split('<response>')[-1].split('<grounding>')[0].strip()
        result_utterance.append({
            'id': id_,
            'utterance': utterance,
            'grounding': grounding
        })
        result_grounding.append({
            "id": id_,
            "no_answer_probability": 0,
            "prediction_text": grounding
        })

with open('utterance.json', 'w') as f:
    json.dump(result_utterance, f, indent=4)

with open('grounding.json', 'w') as f:
    json.dump(result_grounding, f, indent=4)
