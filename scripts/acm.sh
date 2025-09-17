#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

cd ../
  CUDA_VISIBLE_DEVICES=0 \
	python main_full_batch.py \
		--dataset acm --encoder gat --decoder mlp --seed 47 --device cuda \
		--lr 0.0022 --max_epoch 100 \
		--E_para 10 --D_para 10 \
		--loss_E_A_para 1 --loss_E_Z_para 250 --loss_E_H_para 0.5 --decoder_AS_type mean \
		--loss_D_A_para 0.1 --loss_D_H_para 0.9\
	  --missing_rate 0.6 --hyperbuild 2\
		--num_head 2
