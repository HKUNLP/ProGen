#!/bin/bash
export PYTHONPATH=.:${PYTHONPATH}
task_name=sst-2

limit=10000

small_model_name=distilbert-base-uncased
num_epochs=3
learning_rate=2e-5
no_train=false  # whether to train the small model or directly use the checkpoint to validate

# when using synthetic dataset (paper Fig.3), we predict the "gold" labels
# using a oracle model (e.g., roberta trained on gold dataset)
# validation set is also noisy
dataset=
small_model_ckpt=

# otherwise, we experiment on synthetic dataset (paper Fig.6)
exp_noise=true
noise_rate=0.4
noise_val=true  # whether to use a noisy validation set

# the loss used for validation set, ce or rce
obj=ce

output_dir=${small_model_name}-Ns${exp_noise}${noise_rate}Val${noise_val}-p1

cmd="python scripts/IF_exp.py \
--task_name ${task_name} \
--small_model_name ${small_model_name} \
--output_dir ${output_dir} \
--num_epochs ${num_epochs} \
--learning_rate ${learning_rate} \
--obj ${obj} "

if [ "${dataset}" != "" ] ; then
    cmd+=" --dataset out-${task_name}-x2/${dataset}/${task_name}-dataset.jsonl"
fi

if [ "${limit}" != "" ] ; then
  cmd+=" --limit ${limit}"
fi

if [ "${no_train}" = true ] ; then
  cmd+=" --no_train"
fi

if [ "${small_model_ckpt}" != "" ] ; then
  cmd+=" --small_model_ckpt ${small_model_ckpt}"
fi

if [ "${exp_noise}" = "true" ]; then
  cmd+=" --exp_noise --noise_rate ${noise_rate}"
fi

if [ "${noise_val}" = "true" ]; then
  cmd+=" --noise_val"
fi

echo ${cmd}
eval ${cmd}
