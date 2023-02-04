# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to generate datasets.
"""

import argparse
import json
import os

import torch
import wandb

from cls_generator import DataGenerator
from generation import ModelWrapper
from tasks import *
from utils import init_logging, set_seed, read_jsonl, save_jsonl, C_KEY


def create_output_name(args):
    name = [args.model_name.split('/')[-1], args.task_name, f"topk{args.top_k}", f"topp{args.top_p}",
            f"temp{args.temperature}", args.instruction_file.split('/')[-1][:-5]]

    if args.in_context_type != 'none' and args.in_context_num > 0:
        tmp = f"InConType-{args.in_context_type}-{args.in_context_num}-Feed{args.feedback_ratio}"
        if args.keep_mapping:
            tmp += "-Mapping"
        if args.same_y:
            tmp += "-SameY"
        if args.mix_y:
            tmp += f"-MixY{args.order_type}"
        if args.in_context_type.endswith("helpful"):
            tmp += f"-{args.small_model_name}-Hp{args.in_context_ratio}"
        elif args.in_context_type.endswith("harmful"):
            tmp += f"-{args.small_model_name}-Hm{args.remove_ratio}"
        if args.remove_harmful:
            tmp += f"-Rm"

        name.append(tmp)
    return '_'.join(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory to which the generated dataset is saved")
    parser.add_argument("--task_name", type=str, required=True,
                        help="task name")
    parser.add_argument("--instruction_file", type=str, required=True,
                        help="A json file providing the instructions for dataset generation. ")
    parser.add_argument("--exec_type", type=str, default="gx", choices=["p", "gc", "gx"],
                        help="generation type, p: prompting, gc: generation condition C, gx: generation text X")

    # Dataset and prompt parameters
    parser.add_argument("--input_file", type=str, default=None,
                        help="File that contains condition C")
    parser.add_argument("--calibrate", action='store_true',
                        help="Whether to perform calibration in prompting")

    # Text generation and sampling parameters
    parser.add_argument("--model_name", type=str, default="gpt2-xl",
                        help="The pretrained model to use for dataset generation.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="The batch size for generation (only if --input_file is not set)")
    parser.add_argument("--num_entries_per_input", type=int, default=None,
                        help="The number of entries to generate for each label (only if --input_file is not set)")
    parser.add_argument("--max_length", type=int, default=40,
                        help="The maximum output length for each generated text.")
    parser.add_argument("--min_length", type=int, default=10,
                        help="Min length of generated text.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="p value for top-p sampling (set to 0 to perform no top-p sampling)")
    parser.add_argument("--top_k", type=int, default=0,
                        help="k value for top-k sampling (set to 0 to perform no top-k sampling)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="The value used to module the next token probabilities.")
    parser.add_argument("--limit", type=int, default=None,
                        help="The number of instances in the generated dataset.")

    # in-context example parameters
    parser.add_argument("--in_context_type", type=str, default='none',
                        choices=['none', 'val', 'train', 'syn-rand', 'syn-helpful', 'syn-harmful'],
                        help="In context examples to use, syn-* denotes generated examples.")
    parser.add_argument("--in_context_num", type=int, default=0,
                        help="Number of in-context examples")
    parser.add_argument("--keep_mapping", action='store_true',
                        help="Each in-context example keeps the x->y mapping as the original dataset")
    parser.add_argument("--same_y", action='store_true',
                        help="Each X shares the same label")
    parser.add_argument("--same_c", action='store_true',
                        help="Each X shares the same condition text")
    parser.add_argument("--mix_y", action='store_true',
                        help="50% examples have label 0, while other 50% have label 1")
    parser.add_argument("--order_type", type=int, default=1,
                        help="how to arrange examples, only used when mix_y=True. "
                             "type 1: random order"
                             "type 2: label0, label1, label0, label1..."
                             "type 3: N/2 neg, N/2 pos, pos: "
                             "type 4: N/2 pos, N/2 neg, pos: ")
    parser.add_argument("--in_context_ratio", type=float, default=1,
                        help="maximum size of in-context example pool is 10000*ratio")
    parser.add_argument("--remove_ratio", type=float, default=0,
                        help="maximum size of removed example is 10000*ratio")
    parser.add_argument("--feedback_ratio", type=float, default=0,
                        help="ratio to sample from full dataset")
    parser.add_argument("--remove_harmful", action='store_true',
                        help="whether to remove harmful example")

    # Small model parameters
    parser.add_argument("--log_every", type=int, default=10000,
                        help="Train the small model after generating log_every examples.")
    parser.add_argument("--small_model_name", type=str, default='distilbert-base-uncased',
                        help="The small Transformer language model to use.")
    parser.add_argument("--small_model_ckpt", type=str, default=None,
                        help="The saved model to load.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs to train the small model.")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Size of batch to train the small model.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate to train the small model.")

    # Miscellaneous parameters
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.instruction_file, 'r', encoding='utf8') as fh:
        instructions = json.load(fh)
    args.instructions = instructions

    if args.exec_type == "gx":
        output_name = create_output_name(args)
        args.output_dir = os.path.join(args.output_dir, output_name)
        wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"), config=args, name=output_name,
                   tags=[args.task_name])

    logging = init_logging(log_file=args.output_dir + '/output.log', stdout=True)
    logging.info(f"Parameters: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    logging.info("Building downstream classification model...")
    processor = TCProcessor(task_name=args.task_name,
                            model_name=args.small_model_name,
                            model_ckpt=args.small_model_ckpt,
                            output_dir=args.output_dir,
                            device=device,
                            num_epochs=args.num_epochs,
                            train_batch_size=args.train_batch_size,
                            learning_rate=args.learning_rate
                            )

    logging.info("Building generation model...")
    # qa tasks need to generate answer in prompting setting, so we also pad on left
    model = ModelWrapper(model_name=args.model_name, pad_trunc_right=args.exec_type == "p")

    logging.info("Building dataset generator...")
    generator = DataGenerator(
        task_name=args.task_name, exec_type=args.exec_type,
        instructions=instructions, model=model, max_length=args.max_length, min_length=args.min_length,
        top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, do_sample=True,
        processor=processor,
        limit=args.limit,
        in_context_type=args.in_context_type,
        in_context_num=args.in_context_num,
        same_y=args.same_y,
        mix_y=args.mix_y,
        order_type=args.order_type,
        same_c=args.same_c,
        keep_mapping=args.keep_mapping,
        remove_harmful=args.remove_harmful,
        in_context_ratio=args.in_context_ratio,
        remove_ratio=args.remove_ratio,
        feedback_ratio=args.feedback_ratio,
        output_dir=args.output_dir
    )

    if args.exec_type == "p":
        logging.info("Starting prompting inference under zero-shot setting...")
        dataset = processor.dataset[processor.validation_key]
        generator.prompting_inference(dataset, args.batch_size, args.calibrate)
    else:
        if args.input_file:
            logging.info(f"Use condition c from {args.input_file}")
            inputs = [i[C_KEY] for i in read_jsonl(args.input_file)]
        elif args.exec_type == "gx" and processor.sentence2_key is not None:
            logging.info("Use condition c from validation dataset")
            inputs = processor.dataset[processor.validation_key][processor.sentence1_key]
        else:
            logging.info("Do not use condition c")
            inputs = None

        logging.info("Starting dataset generation...")
        outputs = generator.generate_dataset(inputs, num_entries_per_input=args.num_entries_per_input,
                                             batch_size=args.batch_size, log_every=args.log_every)

        logging.info(f"Dataset generation complete, dataset contains {len(outputs)} entries")
        dataset_path = os.path.join(args.output_dir, f'{args.task_name}-dataset.jsonl')
        save_jsonl(outputs, dataset_path)
        logging.info(f"Done saving dataset to file '{dataset_path}'")

    if args.exec_type == "gx":
        wandb.save(args.output_dir)
