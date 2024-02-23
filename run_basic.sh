#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
nohup python -u scripts/trte_deno/train.py --dispatch --start 0 --end 4 > train_0_4.txt &

export CUDA_VISIBLE_DEVICES=1
nohup python -u scripts/trte_deno/train.py --dispatch --start 4 --end 8 > train_4_8.txt &

export CUDA_VISIBLE_DEVICES=2
nohup python -u scripts/trte_deno/train.py --dispatch --start 8 --end 12 > train_8_12.txt &

export CUDA_VISIBLE_DEVICES=3
nohup python -u scripts/trte_deno/train.py --dispatch --start 12 --end 16 > train_12_16.txt &

export CUDA_VISIBLE_DEVICES=4
nohup python -u scripts/trte_deno/train.py --dispatch --start 16 --end 20 > train_16_20.txt &

export CUDA_VISIBLE_DEVICES=5
nohup python -u scripts/trte_deno/train.py --dispatch --start 20 --end 24 > train_20_24.txt &

export CUDA_VISIBLE_DEVICES=6
nohup python -u scripts/trte_deno/train.py --dispatch --start 24 --end 28 > train_24_28.txt &

export CUDA_VISIBLE_DEVICES=7
nohup python -u scripts/trte_deno/train.py --dispatch --start 28 --end 30 > train_28_30.txt &

