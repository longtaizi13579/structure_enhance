import re, string
from datasets import load_metric
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def get_scores(prediction, golden):
    hypos = [x.strip() for x in prediction]
    answers = [[x.strip()] for x in golden]

    f1 = total = 0
    for prediction, ground_truths in zip(hypos, answers):
        total += 1
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    f1 = 100.0 * f1 / total

    metric = load_metric("sacrebleu")
    metric.add_batch(predictions=hypos, references=answers)
    sacrebleu = metric.compute()["score"]

    metric_rouge = load_metric("rouge")
    results = metric_rouge.compute(predictions=hypos, references=answers)
    rougel = results["rougeL"].mid.fmeasure * 100

    return f1, sacrebleu, rougel


if __name__ == '__main__':
    import json
    import utils
    import sacrebleu
    from rouge import Rouge

    language = 'en'
    split = 'test'
    targets = []
    outputs = []
    if language == 'ch':
        with open('../data/Doc2Bot-small-nps/doc2bot-test-kilt.jsonl') as f:
            for line in f.readlines():
                sample = json.loads(line)
                targets.append(sample['output'][0]['answer'])

        # with open('../data/dev_test_result.jsonl') as f:
        with open('../data/doc2bot-test.jsonl') as f:
            for line in f.readlines():
                sample = json.loads(line)
                outputs.append(sample['predictions'][0])

        result_dict = {
            'outputs': outputs,
            'targets': targets[:len(outputs)]
        }
    else:
        with open(f'../data/multidoc2dial/{split}-kilt.jsonl') as f:
            for line in f.readlines():
                sample = json.loads(line)
                targets.append(sample['output'][0]['answer'])

        # with open('../data/dev_test_result.jsonl') as f:
        with open(f'../data/multidoc2dial/{split}_result.jsonl') as f:
            for line in f.readlines():
                sample = json.loads(line)
                outputs.append(sample['predictions'][0])

        result_dict = {
            'outputs': outputs,
            'targets': targets[:len(outputs)]
        }

    if language == 'ch':
        hypothesis_list = [x.replace(' ', '').split('<系统说>：')[-1].strip() for x in result_dict['outputs']]
        hypothesis_list = [x if x else '@' for x in hypothesis_list]
        reference_list = [x.split('<系统说>：')[-1].strip() for x in result_dict['targets']]
        hypothesis_list = [' '.join(x) for x in hypothesis_list]
        reference_list = [' '.join(x) for x in reference_list]
    else:
        hypothesis_list = [x.split('<response>')[-1].strip() for x in result_dict['outputs']]
        hypothesis_list = [x if x else '@' for x in hypothesis_list]
        reference_list = [x.split('<response>')[-1].strip() for x in result_dict['targets']]
        # get_scores(hypothesis_list, reference_list)
    instance_num = len(hypothesis_list)
    f1, em = utils.matching_evaluate(reference_list, hypothesis_list)

    # SacreBleu
    bleu_score = [
        sacrebleu.sentence_bleu(hypothesis, [reference]).score
        for hypothesis, reference in zip(hypothesis_list, reference_list)
    ]
    bleu_score = sum(bleu_score) / instance_num

    # Rouge-L
    rouge_func = Rouge()
    rouge_score = [x['rouge-l']['f'] for x in rouge_func.get_scores(hypothesis_list, reference_list)]
    rouge_score = (sum(rouge_score) / instance_num) * 100

    # f1, bleu_score, rouge_score = get_scores(hypothesis_list, reference_list)

    # f1, sacrebleu, rougel = get_scores(hypothesis_list, reference_list
    print(f'F1:{format(f1, ".2f")} \t S-BLEU:{format(bleu_score, ".2f")} \t Rouge-L:{format(rouge_score, ".2f")}')
