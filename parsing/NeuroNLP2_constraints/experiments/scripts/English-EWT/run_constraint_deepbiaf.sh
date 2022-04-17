#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python -u parsing.py --mode train --config configs/parsing/constraint_biaffine.json --num_epochs 50 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 \
 --word_embedding glove --word_path "/home/jiangnanhugo/PycharmProjects/wilson/dataset/word_embd/glove.6B.100d.txt.gz" --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train "/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_English-EWT/en_ewt-ud-train.conllu" \
 --dev "/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_English-EWT/en_ewt-ud-dev.conllu" \
 --test "/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_English-EWT/en_ewt-ud-test.conllu" \
 --const_file "/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_English-EWT/upos.heads.json"\
 --const_relation_file "/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_English-EWT/upos.relation.json"\
 --model_path "result/constraint_biaffine/"
