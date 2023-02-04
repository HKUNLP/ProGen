#!/bin/bash
export WANDB_DISABLED=true # disable wandb in huggingface transformers
export WANDB_PROJECT=ProGen  # change if needed
export WANDB_ENTITY=  # change to your wandb account
export WANDB_API_KEY=  # change to your api-key
export WANDB_START_METHOD=thread

model_name=gpt2-xl

#model_name=opt175b
#export OPT_LOCAL_URL="http://10.140.1.59:6010/completions"  # your local OPT address

batch_size=32 # for generation with PLM
train_batch_size=32  # for train the small model (DistilBERT by default)

limit=200000
seed=42

task=sst-2
prompt=mv_cls2/p3

in_context_num=8
in_context_type=syn-helpful
feedback_ratio=0.5

# important prompt setting
keep_mapping=true
same_y=true
mix_y=false
order_type=1
same_c=false
in_context_ratio=50

# for lstm
#small_model_name=lstm
#num_epochs=20
#learning_rate=1e-3

# for distilbert
small_model_name=distilbert-base-uncased
num_epochs=3
learning_rate=2e-5

################################################################
for task in sst-2
do
  echo "############################################# Supervised with Human Annotations ###################################################"
  cmd="python3 scripts/misc.py \
  --task_name ${task} \
  --small_model_name ${small_model_name} \
  --train_batch_size ${train_batch_size}
  "
  echo ${cmd}
#  eval ${cmd}


  echo "############################################# Prompting with PLM (Zero-shot performance) ###################################################"
  cmd="python3 main.py \
  --model_name ${model_name} \
  --output_dir out-${task} \
  --task_name ${task} \
  --exec_type p \
  --instruction_file tasks/${prompt}.json \
  --batch_size ${batch_size} \
  --calibrate
  "
  echo ${cmd}
  eval ${cmd}


  echo "############################################# Generating Context C with PLM ###################################################"
  top_k=40
  top_p=0.9
  num_entries_per_input=200000

  cmd="python3 main.py \
   --output_dir out-${task}-x1 \
   --model_name ${model_name} \
   --task_name ${task} \
   --exec_type gc \
   --instruction_file tasks/${prompt}.json \
   --num_entries_per_input ${num_entries_per_input} \
   --top_k ${top_k} \
   --top_p ${top_p} \
   --batch_size 512 \
   --max_length 10 \
   --min_length 1"

  echo ${cmd}
  eval ${cmd}


  echo "############################################# Generating X with PLM ###################################################"
  top_k=0
  top_p=0.9
  temperature=1
  num_entries_per_input=2
  log_every=100000

  cmd="python3 main.py \
   --model_name ${model_name} \
   --output_dir out-${task}-x2 \
   --task_name ${task} \
   --exec_type gx \
   --instruction_file tasks/${prompt}.json \
   --num_entries_per_input ${num_entries_per_input} \
   --batch_size ${batch_size} \
   --train_batch_size ${train_batch_size} \
   --top_k ${top_k} \
   --top_p ${top_p} \
   --small_model_name ${small_model_name} \
   --learning_rate ${learning_rate} \
   --num_epochs ${num_epochs} \
   --min_length 10 \
   --max_length 100 \
   --log_every ${log_every} \
   --limit ${limit} \
   --in_context_type ${in_context_type} \
   --in_context_num ${in_context_num} \
   --in_context_ratio ${in_context_ratio} \
   --feedback_ratio ${feedback_ratio} \
   --seed ${seed} \
   --temperature ${temperature}
   "
  if [ "${same_y}" = "true" ]; then
    cmd+=" --same_y"
  fi
  if [ "${same_c}" = "true" ]; then
    cmd+=" --same_c"
  fi
  if [ "${keep_mapping}" = "true" ]; then
    cmd+=" --keep_mapping"
  fi
  if [ "${mix_y}" = "true" ]; then
    cmd+=" --mix_y --order_type ${order_type}"
  fi
  if [ "${task}" = "sst-2" ] || [ "${task}" = "yelp" ] || [ "${task}" = "elec" ]; then
    cmd+=" --input_file out-${task}-x1/${task}-dataset.jsonl"
  fi

  echo ${cmd}
  eval ${cmd}
done
