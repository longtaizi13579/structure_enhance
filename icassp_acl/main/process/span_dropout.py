import json
import random
import re
from collections import defaultdict


def load_data(datapath):
    result = []
    path = f'{datapath}/multidoc2dial_doc.json'
    with open(path, 'r') as f:
        data = json.load(f)
        for domain in data["doc_data"]:
            for doc_id in data["doc_data"][domain]:
                result.append({
                    "domain": domain,
                    "doc_id": doc_id,
                    "title": data["doc_data"][domain][doc_id]["title"],
                    "doc_text": data["doc_data"][domain][doc_id]["doc_text"],
                    "spans": [
                        {
                            "id_sp": data["doc_data"][domain][doc_id]["spans"][i]["id_sp"],
                            "tag": data["doc_data"][domain][doc_id]["spans"][i]["tag"],
                            "start_sp": data["doc_data"][domain][doc_id]["spans"][i]["start_sp"],
                            "end_sp": data["doc_data"][domain][doc_id]["spans"][i]["end_sp"],
                            "text_sp": data["doc_data"][domain][doc_id]["spans"][i]["text_sp"],
                            "title": data["doc_data"][domain][doc_id]["spans"][i]["title"],
                            "parent_titles": data["doc_data"][domain][doc_id]["spans"][i]["parent_titles"],
                            "id_sec": data["doc_data"][domain][doc_id]["spans"][i]["id_sec"],
                            "start_sec": data["doc_data"][domain][doc_id]["spans"][i]["start_sec"],
                            "text_sec": data["doc_data"][domain][doc_id]["spans"][i]["text_sec"],
                            "end_sec": data["doc_data"][domain][doc_id]["spans"][i]["end_sec"],
                        }
                        for i in data["doc_data"][domain][doc_id]["spans"]
                    ],
                    "doc_html_ts": data["doc_data"][domain][doc_id]["doc_html_ts"],
                    "doc_html_raw": data["doc_data"][domain][doc_id]["doc_html_raw"],
                })
    return result


def text2line(text):
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()


def split_text_section(spans, title):
    def get_text(buff, title, span):
        text = " ".join(buff).replace("\n", " ")
        parent_titles = [title.replace("/", "-").rsplit("#")[0]]
        if len(span["parent_titles"]["text"]) > 1:
            parent_titles = [ele.replace("/", "-").rsplit("#")[0] for ele in span["parent_titles"]["text"]]
        text = " / ".join(parent_titles) + " // " + text
        return text2line(text)

    buff = []
    id_buff = []
    pre_sec, pre_title, pre_span = None, None, None
    passages = []
    span_id_list = []
    subtitles = []
    for span in spans:
        parent_titles = title
        span["parent_titles"] = {
            "id_sp": [x['id_sp'] for x in span["parent_titles"]],
            "text": [x['text'] for x in span["parent_titles"]],
            "level": [x['level'] for x in span["parent_titles"]],
        }
        if len(span["parent_titles"]["text"]) > 1:
            parent_titles = [ele.replace("/", "-").rsplit("#")[0] for ele in span["parent_titles"]["text"]]
            parent_titles = " / ".join(parent_titles)
        if pre_sec == span["id_sec"] or pre_title == span["title"].strip():
            buff.append(span["text_sp"])
            id_buff.append(span['id_sp'])
        elif buff:
            text = get_text(buff, title, pre_span)
            passages.append(text)
            span_id_list.append(id_buff)
            subtitles.append(parent_titles)
            buff = [span["text_sp"]]
            id_buff = [span['id_sp']]
        else:
            buff.append(span["text_sp"])
            id_buff.append(span['id_sp'])
        pre_sec = span["id_sec"]
        pre_span = span
        pre_title = span["title"].strip()
    if buff:
        text = get_text(buff, title, span)
        passages.append(text)
        span_id_list.append(id_buff)
        subtitles.append(parent_titles)
    return passages, subtitles, span_id_list


def process_passage(doc_dataset):
    # doc_dataset = load_dataset("./doc2dial_pub.py", "document_domain", split="train", ignore_verifications=True)
    d_doc_data = defaultdict(dict)  # doc -> "doc_text", "spans"
    d_doc_psg = {}
    doc_psg_all = []
    doc_span_all = []
    doc_titles_all = []
    doc_ids_all = []
    doc_domain_all = []
    d_pid_domain = {}
    start_idx = 0
    for ex in doc_dataset:
        passages, subtitles, span_id_list = split_text_section(ex["spans"], ex["title"])
        doc_psg_all.extend(passages)
        doc_span_all.extend(span_id_list)
        doc_titles_all.extend(subtitles)
        doc_domain_all.extend([ex["domain"]] * len(passages))
        doc_ids_all.extend([ex["doc_id"]] * len(passages))
        d_doc_psg[ex["doc_id"]] = (start_idx, len(passages))
        for i in range(start_idx, start_idx + len(passages)):
            d_pid_domain[i] = ex["domain"]
        start_idx += len(passages)
        d_doc_data[ex["doc_id"]]["doc_text"] = ex["doc_text"]
        d_doc_data[ex["doc_id"]]["spans"] = {}
        d_doc_data[ex["doc_id"]]["domain"] = ex["domain"]
        for d_span in ex["spans"]:
            d_doc_data[ex["doc_id"]]["spans"][d_span["id_sp"]] = d_span

    result = []
    span_passage_map = {}
    psg_span_map = {}
    for domain, doc_id, span_ids, psg in zip(doc_domain_all, doc_ids_all, doc_span_all, doc_psg_all):
        psg_span_map[psg] = []
        for span in span_ids:
            span_passage_map['-'.join([domain, doc_id, span])] = psg
            psg_span_map[psg].append([domain, doc_id, span])

        if psg not in result:
            result.append(psg)

    psg_id_map = {}
    for i in range(len(result)):
        psg_id_map[result[i]] = str(i)

    return span_passage_map, psg_id_map, psg_span_map


def dropout_dd(datapath):
    result = []
    with open(f'{datapath}/train_all_ranker_results_large.json', 'r') as f:
        result += json.load(f)

    for split in ['dev', 'train']:
        with open(f'{datapath}/{split}_span_level.json', 'r') as f:
            dialogs = json.load(f)

        rank_map = {}
        with open(f'{datapath}/whole_{split}.jsonl', 'r') as f:
            for line in f.readlines():
                sample = json.loads(line)
                rank_map[sample['id']] = sample['pid']

        for dialog in dialogs:
            ranks = rank_map[dialog['id']]
            grounding = [str(x) for x in zip(dialog['grounding_passage_index_list'], dialog['passage_span_index_list'])]
            new_ctxs = []
            for psg_index in ranks:
                dropped_all_spans = []
                all_spans = dialog['corresponding_span_list'][psg_index]
                for i in range(len(all_spans)):
                    if str((psg_index, i)) in grounding:
                        dropped_all_spans.append(all_spans[i])
                    elif random.random() > 0.25:
                        dropped_all_spans.append(all_spans[i])
                if dropped_all_spans:
                    new_ctxs.append(' '.join(dropped_all_spans))

            result.append({
                "id": dialog['id'],
                "question": dialog['question'],
                "dialog_act": dialog['dialog_act'],
                "response": dialog['response'],
                "grounding": dialog['grounding'],
                "ctxs": new_ctxs
            })

    return result


def dropout_mdd(datapath):
    path = f'{datapath}/multidoc2dial_doc.json'
    with open(path, 'r') as f:
        doc_data = json.load(f)['doc_data']

    id_grounding_map = {}
    for dialog_file in [f'{datapath}/multidoc2dial_dial_train.json',
                        f'{datapath}/multidoc2dial_dial_validation.json']:
        with open(dialog_file, 'r') as f:
            dial_data = json.load(f)['dial_data']
            for domain, dialogs in dial_data.items():
                for dialog in dialogs:
                    dialog_id = dialog['dial_id']
                    for turn in dialog['turns']:
                        turn_id = dialog_id + '_' + str(turn['turn_id'])
                        id_grounding_map[turn_id] = []
                        for grounding in turn['references']:
                            id_grounding_map[turn_id].append(
                                '-'.join([domain, grounding['doc_id'], grounding['id_sp']]))

    span_passage_map, psg_id_map, psg_span_map = process_passage(load_data(datapath))
    new_psg_span_map = {}
    for k, v in psg_span_map.items():
        new_psg_span_map[re.sub(' +', ' ', k)] = v
    psg_span_map = new_psg_span_map

    with open(f'{datapath}/multi_all_v1_ranker.json', 'r') as f:
        samples = json.load(f)

    for sample in samples:
        id_ = sample['id']
        ctxs = sample['ctxs']
        new_ctxs = []
        groundings = id_grounding_map[id_]
        for ctx in ctxs:
            ctx_buffer = []
            all_spans = psg_span_map[ctx]
            for span in all_spans:
                if '-'.join(span) in groundings:
                    ctx_buffer.append(doc_data[span[0]][span[1]]['spans'][span[2]]['text_sp'])
                elif random.random() > 0.25:
                    ctx_buffer.append(doc_data[span[0]][span[1]]['spans'][span[2]]['text_sp'])
            if ctx_buffer:
                title = ctx.split('//')[0]
                new_ctxs.append(text2line(title + '//' + ' '.join(ctx_buffer)))
        sample['ctxs'] = new_ctxs

    with open(f'{datapath}/multi_all_v1_ranker.json', 'r') as f:
        old_samples = json.load(f)

    return old_samples + samples
