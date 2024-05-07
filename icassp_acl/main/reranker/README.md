# Reranker

## LongFormer

### Single-gpu Train

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model allenai/longformer-base-4096 \
        --name longformer_base_v1 \
        --code longformer \
        --mode train \
        --epochs 5 \
        --per_gpu_batch_size 3 \
        --accumulation_steps 1 \
        --warmup_steps 500 \
        --eval_batch_size 2 \
        --lr 3e-5 \
        --eps 1e-06 \
        --source_sequence_size 4096 \
        --dataset longformer-train \
        --passages 20 \
        --gradient_checkpoint_segments 5
```


## TreeJC

### Single-gpu Train

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model roberta-base \
        --name treejc_base_v1 \
        --code treejc \
        --mode train \
        --epochs 5 \
        --per_gpu_batch_size 3 \
        --accumulation_steps 1 \
        --warmup_steps 500 \
        --eval_batch_size 2 \
        --lr 3e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --dataset treejc-train \
        --passages 20
```