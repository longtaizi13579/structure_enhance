# MultiDoc2Dial

## Bart

## UniGDD

### Single-gpu Train

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model t5-base \
        --name unigdd \
        --code unigdd \
        --mode train \
        --n_gpu 1 \
        --epochs 10 \
        --per_gpu_batch_size 2 \
        --accumulation_steps 4 \
        --warmup_steps 500 \
        --eval_batch_size 16 \
        --lr 1e-4 \
        --eps 1e-06 
```

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model t5-large \
        --name fid \
        --code fid \
        --mode train \
        --epochs 5 \
        --per_gpu_batch_size 1 \
        --accumulation_steps 8 \
        --warmup_steps 500 \
        --eval_batch_size 4 \
        --lr 1e-4 \
        --eps 1e-06 \
        --max_grad_norm 1 \
        --weight_decay 0.0 \
        --source_sequence_size 512 \
        --target_sequence_size 400 \
        --dataset mdd-train \
        --debug
```


```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --name mt5_workshop_v1 \
        --code mt5 \
        --mode train \
        --epochs 10 \
        --per_gpu_batch_size 4 \
        --accumulation_steps 4 \
        --warmup_steps 500 \
        --eval_batch_size 16 \
        --lr 1e-4 \
        --eps 1e-06 \
        --max_grad_norm 1 \
        --weight_decay 0.0 \
        --source_sequence_size 512 \
        --target_sequence_size 512 \
        --dataset workshop \
        --reload
```

### Multi-gpu Train

```shell
export OMP_NUM_THREADS=1

/root/data/huairang/Doc2dial/python/bin/python -m torch.distributed.launch --nproc_per_node 8 \
        /root/data/huairang/Doc2dial-Baseline/main.py \
        --home /root/data/huairang/Doc2dial/ \
        --model t5-3b \
        --name unigdd-3b-multi \
        --code unigdd \
        --mode train \
        --n_gpu 8 \
        --epochs 10 \
        --per_gpu_batch_size 4 \
        --accumulation_steps 2 \
        --warmup_steps 500 \
        --eval_batch_size 16 \
        --lr 2e-4 \
        --eps 1e-06 \
        --no_find_unused_parameters 
```

### Valid

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model t5-base \
        --name unigdd \
        --code unigdd \
        --mode valid \
        --n_gpu 1 \
        --eval_batch_size 16 
```

### Test

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --model t5-large \
        --name dp4_multi_t5_large_v2 \
        --code unigdd \
        --mode test \
        --eval_batch_size 2 \
        --source_sequence_size 3000 \
        --target_sequence_size 400 \
        --dataset mdd-dev-test \
        --beam_size 2
```