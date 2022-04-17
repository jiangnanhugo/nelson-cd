#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u parsing.py --mode train --config configs/parsing/biaffine.json --num_epochs 100 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 \
 --word_embedding glove --word_path "/home/jiangnanhugo/PycharmProjects/wilson/dataset/word_embd/glove.6B.100d.txt.gz" --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train "/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_English-GUM/en_gum-ud-train.conllu" \
 --dev "/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_English-GUM/en_gum-ud-dev.conllu" \
 --test "/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_English-GUM/en_gum-ud-test.conllu" \
 --const_file "/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_English-GUM/inv_dependency.json"\
 --model_path "result/deepbiaf/English-GUM"
