#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
nohup python -u scripts/trte_deno/train.py --dispatch --start 0 --end 1 > train_0.txt &

# export CUDA_VISIBLE_DEVICES=1
# nohup python -u scripts/trte_deno/train.py --dispatch --start 1 --end 2 > train_1.txt &

# export CUDA_VISIBLE_DEVICES=2
# nohup python -u scripts/trte_deno/train.py --dispatch --start 2 --end 3 > train_2.txt &

# export CUDA_VISIBLE_DEVICES=3
# nohup python -u scripts/trte_deno/train.py --dispatch --start 3 --end 4 > train_3.txt &

# export CUDA_VISIBLE_DEVICES=4
# nohup python -u scripts/trte_deno/train.py --dispatch --start 4 --end 5 > train_4.txt &

