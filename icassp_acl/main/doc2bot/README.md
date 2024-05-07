# Dialog Policy Learning

## BERT

### Single-gpu Train

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model bert \
        --name bert_base_v1 \
        --code dpl \
        --mode train \
        --epochs 5 \
        --per_gpu_batch_size 32 \
        --accumulation_steps 1 \
        --warmup_steps 500 \
        --eval_batch_size 40 \
        --lr 2e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --dataset doc2bot \
        --debug
```

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model bert \
        --name bert_base_struct_v1 \
        --code dpl \
        --mode train \
        --epochs 5 \
        --per_gpu_batch_size 32 \
        --accumulation_steps 1 \
        --warmup_steps 500 \
        --eval_batch_size 40 \
        --lr 2e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --dataset doc2bot-rm-struct \
        --rm_struct \
        --debug
```

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model bert \
        --name bert_act_v1 \
        --code act \
        --mode train \
        --epochs 10 \
        --per_gpu_batch_size 32 \
        --accumulation_steps 1 \
        --warmup_rate 0.05 \
        --eval_batch_size 40 \
        --lr 2e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --dataset doc2bot \
        --debug
```

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model roberta \
        --name roberta_act_v1 \
        --code act \
        --mode train \
        --epochs 10 \
        --per_gpu_batch_size 32 \
        --accumulation_steps 1 \
        --warmup_rate 0.05 \
        --eval_batch_size 40 \
        --lr 2e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --dataset doc2bot
```

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model electra \
        --name electra_act_v1 \
        --code act \
        --mode train \
        --epochs 10 \
        --per_gpu_batch_size 32 \
        --accumulation_steps 1 \
        --warmup_rate 0.05 \
        --eval_batch_size 40 \
        --lr 1e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --dataset doc2bot
```

## RoBERTa

### Single-gpu Train

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model roberta \
        --name roberta_base_v1 \
        --code dpl \
        --mode train \
        --epochs 5 \
        --per_gpu_batch_size 32 \
        --accumulation_steps 1 \
        --warmup_steps 500 \
        --eval_batch_size 40 \
        --lr 2e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --dataset doc2bot \
        --debug
```

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model roberta \
        --name roberta_base_struct_v1 \
        --code dpl \
        --mode train \
        --epochs 5 \
        --per_gpu_batch_size 32 \
        --accumulation_steps 1 \
        --warmup_steps 500 \
        --eval_batch_size 40 \
        --lr 2e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --dataset doc2bot-rm-struct \
        --rm_struct \
        --debug
```


## ELECTRA

### Single-gpu Train

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model electra \
        --name electra_base_v1 \
        --code dpl \
        --mode train \
        --epochs 5 \
        --per_gpu_batch_size 32 \
        --accumulation_steps 1 \
        --warmup_steps 500 \
        --eval_batch_size 40 \
        --lr 1e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --dataset doc2bot \
        --debug
```

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model electra \
        --name electra_base_struct_v1 \
        --code dpl \
        --mode train \
        --epochs 5 \
        --per_gpu_batch_size 32 \
        --accumulation_steps 1 \
        --warmup_steps 500 \
        --eval_batch_size 40 \
        --lr 1e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --dataset doc2bot-rm-struct \
        --rm_struct \
        --debug
```

## Generation
```shell
python  main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --name sfre2g_graph_v1 \
        --rerank_path /mnt/data/huairang/Doc2dial/another_method_common \
        --code sfre2g \
        --mode train \
        --epochs 10 \
        --per_gpu_batch_size 2 \
        --accumulation_steps 4 \
        --warmup_steps 500 \
        --eval_batch_size 8 \
        --lr 2e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --target_sequence_size 512 \
        --dataset doc2bot-act \
        --passages 5 \
        --subgraph \
        --reload \
        --debug
```