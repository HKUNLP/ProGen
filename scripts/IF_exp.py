#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os

import torch
from datasets import load_from_disk

from cls_generator import list_to_hf_dataset
from influence_utils import experiments, misc_utils
from tasks import TCProcessor
from utils import set_seed, read_jsonl
import numpy as np


def corrupt_label(dataset, corrupt_ratio=0.4):
    ori_labels = np.array(dataset['label'])
    corrupt_labels = np.array(ori_labels)
    label0_idx = np.where(ori_labels == 0)[0]
    label1_idx = np.where(ori_labels == 1)[0]
    np.random.shuffle(label0_idx)
    np.random.shuffle(label1_idx)
    label0_idx = label0_idx[:int(len(label0_idx) * corrupt_ratio)]
    label1_idx = label1_idx[:int(len(label1_idx) * corrupt_ratio)]
    corrupt_labels[label1_idx] = 0
    corrupt_labels[label0_idx] = 1
    dataset = dataset.remove_columns(['label'])
    dataset = dataset.add_column('label', corrupt_labels.tolist())
    if_corrupted = np.zeros_like(ori_labels, dtype=bool)
    if_corrupted[label0_idx] = True
    if_corrupted[label1_idx] = True
    return ori_labels, if_corrupted, dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default='sst-2',
                        help="The output directory for storing the trained model and evaluation results.")
    parser.add_argument("--output_dir", type=str, default='tmp',
                        help="The output directory for storing the trained model and evaluation results.")

    # Model and training parameters
    parser.add_argument("--dataset", type=str, default=None,
                        help="Local dataset to use.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use limited samples to use for training.")
    parser.add_argument("--no_train", action='store_true',
                        help="Only evaluate.")
    parser.add_argument("--small_model_name", type=str, default='distilbert-base-uncased',
                        help="The pretrained Transformer language model to use.")
    parser.add_argument("--small_model_ckpt", type=str, default=None,
                        help="The saved model to load.")
    parser.add_argument("--oracle_model_ckpt", type=str, default=None,
                        help="The oracle model to load.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs to train the small model.")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Size of batch to train the small model.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate to train the small model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed used to initialize all random number generators.")
    parser.add_argument("--log_every", type=int, default=None,
                        help="Train the small model after generating log_every examples.")
    parser.add_argument("--exp_noise", action='store_true',
                        help="gold noise experiment")
    parser.add_argument("--noise_rate", type=float, default=0.2,
                        help="noise in train data")
    parser.add_argument("--noise_val", action='store_true',
                        help="whether to add noise in val data")
    parser.add_argument("--obj", type=str, default="rce",
                        help="obj in influence function")

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args.seed)
    print(f"Parameters: {args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = TCProcessor(task_name=args.task_name,
                            model_name=args.small_model_name,
                            model_ckpt=args.small_model_ckpt,
                            output_dir=args.output_dir,
                            device=device,
                            num_epochs=args.num_epochs,
                            train_batch_size=args.train_batch_size,
                            learning_rate=args.learning_rate
                            )

    if args.dataset:  # synthetic dataset
        if os.path.isdir(args.dataset):
            dataset = load_from_disk(args.dataset)
            if args.limit and len(dataset) > args.limit:
                dataset = dataset.select(range(args.limit))
        else:
            dataset = read_jsonl(args.dataset)
            if args.limit:
                dataset = dataset[:args.limit]
            dataset = list_to_hf_dataset(dataset, processor.label_key, processor.sentence1_key, processor.sentence2_key)
    else:  # standard dataset
        dataset = processor.dataset['train']
        print(len(dataset),args.limit)
        if args.limit and len(dataset) > args.limit:
            dataset = dataset.select(range(args.limit))

    if args.dataset:
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset, val_dataset = dataset['train'].flatten_indices(), dataset['test'].flatten_indices()
        if not args.noise_val:  # if not use noise validation set, then use gold val set
            val_dataset = processor.dataset[processor.validation_key]
        encoded_train_dataset, encoded_val_dataset = processor.load_train_val(train_dataset, val_dataset)
    elif args.exp_noise:
        train_dataset, val_dataset = dataset, processor.dataset[processor.validation_key]
        # manually corrupt labels to construct a noisy training dataset
        ori_labels, if_corrupted, train_dataset = corrupt_label(train_dataset, args.noise_rate)
        if args.noise_val:
            # manually corrupt labels to construct a noisy validation dataset
            _, _, val_dataset = corrupt_label(val_dataset, args.noise_rate)
        encoded_train_dataset, encoded_val_dataset = processor.load_train_val(train_dataset, val_dataset)
    else:
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset, val_dataset = dataset['train'].flatten_indices(), dataset['test'].flatten_indices()
        encoded_train_dataset, encoded_val_dataset = processor.load_train_val(train_dataset, val_dataset)

    if args.small_model_ckpt is None:
        processor.train(encoded_train_dataset, encoded_val_dataset, train=not args.no_train)
    val_metric, val_loss = processor.validate()
    logging.info(f"Full gold val dataset, metric: {val_metric['eval_accuracy']}; loss: {val_loss}")

    random_idx = list(range(len(encoded_val_dataset)))
    set_seed(args.seed)
    np.random.shuffle(random_idx)

    init_accs = {}
    init_losses = {}
    for num_examples_to_test in [len(processor.encoded_dataset[processor.validation_key])]:
        # select some examples
        selected = random_idx[:num_examples_to_test]
        if_val_dataset = val_dataset.select(selected)
        encoded_if_val_dataset = encoded_val_dataset.select(selected)

        # initial results
        val_metric, val_loss = processor.validate()
        init_accs[num_examples_to_test] = val_metric['eval_accuracy']
        init_losses[num_examples_to_test] = val_loss

    for num_examples_to_test in [len(processor.encoded_dataset[processor.validation_key])]:
        # select some examples
        selected = random_idx[:num_examples_to_test]
        if_val_dataset = val_dataset.select(selected)
        encoded_if_val_dataset = encoded_val_dataset.select(selected)

        # calculate importance score for each training instance
        outputs_collections = experiments.run_full_influence_functions_dataset(
            model=processor.model,
            train_dataset=encoded_train_dataset,
            val_dataset=encoded_if_val_dataset,
            tokenizer=processor.tokenizer,
            s_test_num_samples=1000,
            s_test_obj=args.obj,
            weight_decay=0.005
        )
        # cache it in case we use it multiple times
        torch.save(outputs_collections, f'{args.output_dir}/full-output-{num_examples_to_test}.pt')

        outputs_collections = torch.load(f'{args.output_dir}/full-output-{num_examples_to_test}.pt')

        # 1. show cases
        if num_examples_to_test == 1:
            print("test sample:", if_val_dataset[0])
            helpful_indices, harmful_indices = misc_utils.get_helpful_harmful_indices_from_influences_dict(
                outputs_collections[-1]['influences'])
            for indices, idx_type in zip([helpful_indices, harmful_indices], ['helpful', 'harmful']):
                print(f"Top 5 {idx_type} instances: ")
                for i in indices[:5]:
                    if args.exp_noise:
                        print(f"instances: {train_dataset[i]}, if score: {outputs_collections[-1]['influences'][i]}, "
                              f"ori_label: {ori_labels[i]}")
                    else:
                        print(f"instances: {train_dataset[i]}, if score: {outputs_collections[-1]['influences'][i]}")

        # 2. remove experiments and draw figure, e.g., Fig 6.
        experiments.run_avg_percent(outputs_collections, init_acc=init_accs[num_examples_to_test],
                                    init_loss=init_losses[num_examples_to_test], processor=processor,
                                    train_dataset=encoded_train_dataset, val_dataset=encoded_val_dataset,
                                    test_dataset=processor.encoded_dataset[processor.validation_key],
                                    save_path=f"{args.output_dir}/{args.obj}-percent-clsb-avg-{num_examples_to_test}.png",
                                    class_balance=True, if_corrupted=if_corrupted if args.exp_noise else None)
