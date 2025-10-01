#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

cd ../
  CUDA_VISIBLE_DEVICES=0 \
	python main_full_batch.py \
		--dataset acm --encoder gat --decoder mlp --seed 47 --device cuda \
		--lr 0.0021 --max_epoch 100 \
		--decoder_AH_type mean \
		--loss_APA_A2H_para 0.1 --loss_APA_H2A_para 0.9\
	  --missing_rate 0.6 --hyperbuild 2\
		--num_head 2
