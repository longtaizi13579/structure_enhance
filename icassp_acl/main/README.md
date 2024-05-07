# Doc2dial Baseline

## Doc2dial

## MultiDoc2Dial

### Bart

config python home

```shell
export PYTHONPATH=$PYTHONPATH:/mnt/workspace/haomin/Doc2dial-Baseline
```

train

```shell
python main.py \
        --code bart \
        --mode train \
        --name experiment \
        --gpu 0 \
        --n_gpu 1 \
        --per_gpu_batch_size 16 \
        --print_freq 40
```

valid

```shell
python main.py \
        --code bart \
        --mode valid \
        --name experiment \
        --gpu 0 \
        --n_gpu 1 \
        --eval_batch_size 64
```

test

```shell
python main.py \
        --code bart \
        --mode test \
        --name experiment \
        --gpu 0 \
        --n_gpu 1 \
        --eval_batch_size 64
```

### UniGDD

multi-gpu train

```shell
/root/data/huairang/Doc2dial/python/bin/python -m torch.distributed.launch --nproc_per_node 8 \
        /root/data/huairang/Doc2dial-Baseline/main.py \
        --home /root/data/huairang/Doc2dial/ \
        --name unigdd \
        --code unigdd \
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

single-gpu train

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
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

valid

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --name unigdd \
        --code unigdd \
        --mode valid \
        --n_gpu 1 \
        --epochs 10 \
        --per_gpu_batch_size 2 \
        --accumulation_steps 4 \
        --warmup_steps 500 \
        --eval_batch_size 16 \
        --lr 1e-4 \
        --eps 1e-06 
```

test

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --name unigdd-821 \
        --code unigdd \
        --mode test \
        --n_gpu 1 \
        --epochs 10 \
        --per_gpu_batch_size 2 \
        --accumulation_steps 4 \
        --warmup_steps 500 \
        --eval_batch_size 16 \
        --lr 1e-4 \
        --eps 1e-06 
```

## Reranker

### Transformer

train

```shell
python main.py \
        --code transformer \
        --mode train \
        --name experiment \
        --gpu 0 \
        --n_gpu 1 \
        --eval_batch_size 64
```

test

```shell
python main.py \
        --code transformer \
        --mode test \
        --name transformer \
        --gpu 0 \
        --n_gpu 1 \
        --eval_batch_size 128
```

### RGCN

multi-gpu train

```shell
python -m torch.distributed.launch --nproc_per_node 4 \
        main.py \
        --code graph \
        --mode train \
        --name span_context \
        --n_gpu 4 \
        --epochs 3 \
        --per_gpu_batch_size 4 \
        --eval_batch_size 128
```

single-gpu train

```shell
python main.py \
        --code graph \
        --mode train \
        --name span_context_nograph \
        --n_gpu 1 \
        --epochs 3 \
        --per_gpu_batch_size 4 \
        --eval_batch_size 128
```

test

```shell
python main.py \
        --code graph \
        --mode test \
        --name experiment \
        --gpu 0 \
        --n_gpu 1 \
        --epochs 3 \
        --per_gpu_batch_size 4 \
        --eval_batch_size 128
```