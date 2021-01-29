#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/tmp/DeBERTa/

function setup_glue_data(){
	task=$1
	mkdir -p $cache_dir
	if [[ ! -e $cache_dir/glue_tasks/${task}/train.tsv ]]; then
		curl -J -L https://raw.githubusercontent.com/nyu-mll/jiant/v1.3.2/scripts/download_glue_data.py | python3 - --data_dir $cache_dir/glue_tasks 
	fi
}

init=large 
init=/mnt/penhe/models/DeBERTa/XLarge_MNLI/pytorch.model_91.4_.bin
tag=Large
Task=RTE
setup_glue_data $Task
data_dir=/mnt/penhe/glue/data/$Task
#../utils/train.sh -i $init --config config.json -t $Task --data ${data_dir} --tag $tag -o /tmp/ttonly/$tag/$Task -- --num_train_epochs 6 --accumulative_update 1 --warmup 100 --learning_rate 6e-6 --train_batch_size 16 --max_seq_length 320 --dump 500 --cls_drop 0.50 --max_grad_norm 1

init_dir=/mnt/penhe/models/SuperGLUE/RTE/RTE_97
init=${init_dir}/pytorch.model-000311.bin
../utils/train.sh -p -i $init --config config.json -t $Task --data ${data_dir} --tag $tag -o ${init_dir}/Pred -- --num_train_epochs 2 --accumulative_update 2 --warmup 100 --learning_rate 7e-6 --train_batch_size 32 --max_seq_length 320 --dump 500 --cls_drop 0.15 --max_grad_norm 1 
