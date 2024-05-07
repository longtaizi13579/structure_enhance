# Universe

## UniGDD

### Single-gpu Train

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --name unigdd \
        --code universe \
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

### Multi-gpu Train

```shell
export OMP_NUM_THREADS=1
/root/data/huairang/Doc2dial/python/bin/python -m torch.distributed.launch --nproc_per_node 8 \
        /root/data/huairang/Doc2dial-Baseline/main.py \
        --home /root/data/huairang/Doc2dial/ \
        --name unigdd \
        --code universe \
        --mode train \
        --n_gpu 8 \
        --epochs 10 \
        --per_gpu_batch_size 4 \
        --accumulation_steps 2 \
        --warmup_steps 500 \
        --eval_batch_size 16 \
        --lr 1e-4 \
        --eps 1e-06 
```

### Valid

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --name unigdd \
        --code universe \
        --mode valid \
        --n_gpu 1 \
        --eval_batch_size 16 
```


### Test

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --name unigdd \
        --code universe \
        --mode test \
        --n_gpu 1 \
        --eval_batch_size 16 
```