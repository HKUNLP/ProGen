#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import torch
from utils import set_seed, read_jsonl, init_logging
import os
from datasets import load_from_disk
from tasks import TCProcessor
from cls_generator import list_to_hf_dataset


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
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs to train the small model.")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Size of batch to train the small model.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate to train the small model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed used to initialize all random number generators.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging = init_logging(log_file=args.output_dir + '/output.log', stdout=True)

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

    if args.dataset is not None:
        if os.path.isdir(args.dataset):
            dataset = load_from_disk(args.dataset)
            if args.limit:
                dataset = dataset.select(range(args.limit))
        else:
            dataset = read_jsonl(args.dataset)
            if args.limit:
                dataset = dataset[:args.limit]
            dataset = list_to_hf_dataset(dataset, processor.label_key, processor.sentence1_key, processor.sentence2_key)
    else:  # standard dataset
        dataset = processor.dataset['train']
        if args.limit:
            dataset = dataset.select(range(args.limit))

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset, val_dataset = dataset['train'], dataset['test']
    encoded_train_dataset, encoded_val_dataset = processor.load_train_val(train_dataset, val_dataset)

    # calculate mauve or self-bleu score
    # import mauve
    # from scripts.self_bleu import cal_self_bleu
    # import numpy as np

    # p_text = processor.dataset[processor.validation_key]['text'][:1000]
    # q_text = np.random.choice(train_dataset['text'][10000:], size=1000, replace=False).tolist()  # IF is not applied in first 10000
    # out = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0, max_text_length=256, verbose=False,
    #                           batch_size=32, num_buckets=30, featurize_model_name='gpt2-xl', mauve_scaling_factor=1)
    # logging.info(f"{args.dataset}, train size: {len(train_dataset)}")
    # logging.info(f"MAUVE score: {out.mauve}")

    # logging.info("gold self-bleu: ")
    # cal_self_bleu(p_text, 1000)
    # logging.info("prediction self-bleu: ")
    # cal_self_bleu(q_text, 1000)

    processor.train(encoded_train_dataset, encoded_val_dataset, train=not args.no_train)
    logging.info("Metric on validation set: " + str(processor.validate()[0]))
    logging.info("Metric on syn set: " + str(processor.validate(encoded_train_dataset)[0]))