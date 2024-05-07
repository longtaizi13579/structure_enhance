# Retrieval

## DPR

### Single-gpu Train

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model hfl/chinese-roberta-wwm-ext \
        --name dpr_concate_base_v1 \
        --code dpr \
        --mode train \
        --epochs 10 \
        --per_gpu_batch_size 128 \
        --accumulation_steps 1 \
        --warmup_rate 0.1 \
        --eval_batch_size 128 \
        --lr 2e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --gradient_checkpoint_segments 64 \
        --dataset_prefix concate- \
        --add_parent
```


### Single-gpu Test

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model hfl/chinese-roberta-wwm-ext \
        --name dpr_base_v1 \
        --code dpr \
        --mode test \
        --epochs 10 \
        --eval_batch_size 128 \
        --source_sequence_size 512 \
        --debug
```