# Universe

## RAG

### Single-gpu Train

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --name rag_v1 \
        --code rag \
        --mode train \
        --epochs 10 \
        --per_gpu_batch_size 4 \
        --accumulation_steps 4 \
        --warmup_steps 500 \
        --eval_batch_size 8 \
        --lr 3e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --target_sequence_size 512 \
        --dataset rag \
        --passages 5
```

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --language ch \
        --name sfre2g_doc2bot_v1 \
        --rerank_path /root/data/huihuan/re2g_acl/kgi-slot-filling/models/another_method_common \
        --code rag \
        --mode train \
        --epochs 10 \
        --per_gpu_batch_size 4 \
        --accumulation_steps 4 \
        --warmup_steps 500 \
        --eval_batch_size 8 \
        --lr 3e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --target_sequence_size 512 \
        --dataset doc2bot \
        --passages 5 \
        --reload \
        --debug
```


## Struc-FiD

### Single-gpu Train

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --name struc_fid_v1 \
        --code struc-fid \
        --mode train \
        --epochs 10 \
        --per_gpu_batch_size 4 \
        --accumulation_steps 4 \
        --warmup_steps 500 \
        --eval_batch_size 8 \
        --lr 3e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --target_sequence_size 512 \
        --dataset struc_fid \
        --passages 5
```

```shell
python main.py \
        --home /mnt/data/huairang/Doc2dial/ \
        --language ch \
        --name rag_ch_v1 \
        --code rag \
        --mode train \
        --epochs 10 \
        --per_gpu_batch_size 4 \
        --accumulation_steps 4 \
        --warmup_steps 500 \
        --eval_batch_size 8 \
        --lr 3e-5 \
        --eps 1e-06 \
        --source_sequence_size 512 \
        --target_sequence_size 512 \
        --dataset struc_fid \
        --passages 5 
```