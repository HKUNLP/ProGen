#!/bin/bash
export PYTHONPATH=.:${PYTHONPATH}
# the generated dataset to use
dataset=gpt2-xl_sst-2_topk0_topp0.9_temp1.0_p3_InConType-syn-helpful-8-Feed0.5-Mapping-SameY-distilbert-base-uncased-Hp50.0
limit=100000  # limit the instances to load from the above dataset

# for lstm
#small_model_name=lstm
#num_epochs=20
#learning_rate=1e-3

# for distilbert
small_model_name=distilbert-base-uncased
num_epochs=3
learning_rate=2e-5

small_model_ckpt= # load a trained model
no_train=false  # whether to train the small model or directly use the checkpoint (above) to validate

output_dir=out-${task_name}-${small_model_name}
cmd="python scripts/misc.py \
--task_name ${task_name} \
--small_model_name ${small_model_name} \
--output_dir ${output_dir} \
--num_epochs ${num_epochs} \
--learning_rate ${learning_rate} \
--dataset out-${task_name}-x2/${dataset}/${task_name}-dataset.jsonl"

if [ "${limit}" != "" ] ; then
  cmd+=" --limit ${limit}"
fi

if [ "${no_train}" = true ] ; then
  cmd+=" --no_train"
fi

if [ "${small_model_ckpt}" != "" ] ; then
  cmd+=" --small_model_ckpt ${small_model_ckpt}"
fi

echo ${cmd}
eval ${cmd}

