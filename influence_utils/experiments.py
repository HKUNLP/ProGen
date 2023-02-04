from influence_utils import misc_utils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import torch
from typing import *
from influence_utils import influence_helpers
from contexttimer import Timer
from transformers import AutoTokenizer, DataCollatorWithPadding
import logging

RECALL_KS = [10, 100, 1000]
NUM_NEIGHBORS = [10, 100, 1000, 10000]
RECALL_NAMES = ["Most Helpful",
                "Most Harmful",
                "Most Influencetial",
                "Least Influential"]

NUMS = [1, 10, 100, 1000, 5000, 10000]
PERCENT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def plot_distribution(
        distribution: List,
        save_path: str
) -> None:
    plt.hist(distribution)
    plt.savefig(save_path)
    plt.close()


def plot_distributions(
    distributions: Dict,
    save_path: str
) -> None:
    n = len(list(distributions.keys()))
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5), dpi=100, sharex=True, sharey=True)
    for i, (k, v) in enumerate(distributions.items()):
        axes[i].hist(v)
        axes[i].set_title(k)

    fig.savefig(save_path)
    plt.close(fig)


def get_recall(model, example, faiss_index, full_influences_dict):
    features = misc_utils.compute_BERT_CLS_feature(model, example)
    features = features.cpu().detach().numpy()
    if list(full_influences_dict.keys()) != list(range(len(full_influences_dict))):
        raise ValueError

    full_influences = []
    for key in sorted(full_influences_dict):
        full_influences.append(full_influences_dict[key])

    sorted_indices_small_to_large = np.argsort(full_influences)
    sorted_indices_large_to_small = np.argsort(full_influences)[::-1]
    sorted_indices_abs_large_to_small = np.argsort(np.abs(full_influences))[::-1]
    sorted_indices_abs_small_to_large = np.argsort(np.abs(full_influences))

    recalls_collections = {}
    for i, (name, sorted_indices) in enumerate(zip(
            RECALL_NAMES,
            [sorted_indices_small_to_large,
             sorted_indices_large_to_small,
             sorted_indices_abs_large_to_small,
             sorted_indices_abs_small_to_large])):

        recalls_collection = []
        for recall_k in tqdm(RECALL_KS):
            recalls = []
            influential = sorted_indices[:recall_k]
            influential_set = set(influential.tolist())
            for k in NUM_NEIGHBORS:
                distances, indices = faiss_index.search(k=k, queries=features)
                indices_set = set(indices.squeeze(axis=0).tolist())
                recall = len(influential_set & indices_set) / len(influential_set)
                recalls.append(recall)

            recalls_collection.append(recalls)
        recalls_collections[name] = recalls_collection
    return recalls_collections


def plot_recall(recalls_list, save_path):
    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=[20, 5])
    for i, name in enumerate(RECALL_NAMES):
        recalls_collection = [recalls[name] for recalls in recalls_list]
        for j, recall_k in tqdm(enumerate(RECALL_KS)):
            recalls = np.array([item[j] for item in recalls_collection])
            mean_recalls = recalls.mean(axis=0)
            error = recalls.max(axis=0) - mean_recalls
            axes[i].plot(NUM_NEIGHBORS, mean_recalls,
                         linestyle="--", marker="o",
                         label=f"recall@{recall_k}")
            # axes[i].errorbar(NUM_NEIGHBORS, mean_recalls, yerr=error, fmt='--o')

        axes[i].legend()
        axes[i].set_title(name)
        axes[i].set_xscale("log")
        axes[i].set_ylabel("Recall")
        axes[i].set_xlabel("Number of Nearest Neighbors")
    fig.savefig(save_path)
    plt.close(fig)


def run_full_influence_functions(
        model: torch.nn.Module,
        val_dataset: Dataset,
        train_dataset: Dataset,
        tokenizer: AutoTokenizer,
        num_examples_to_test: int,
        s_test_num_samples: int = 1000,
        mode: str = "all"
) -> Dict[int, Dict[str, Any]]:
    if mode not in ["all", "only-correct", "only-incorrect"]:
        raise ValueError(f"Unrecognized mode {mode}")

    eval_instance_data_loader = misc_utils.get_dataloader(dataset=val_dataset,
                                                          batch_size=1,
                                                          random=False)

    num_examples_tested = 0
    outputs_collections = {}
    precomputed_grad_train_dict = {}
    batch_size = min(max(1, len(train_dataset) // s_test_num_samples), 128)

    print(f"using batch_sise = {batch_size} to calculate s_test")
    for test_index, test_inputs in enumerate(eval_instance_data_loader):
        if num_examples_tested >= num_examples_to_test:
            break

        # Skip when we only want cases of correction prediction but the
        # prediction is incorrect, or vice versa
        prediction_is_correct = misc_utils.is_prediction_correct(
            model=model,
            inputs=test_inputs)

        if mode == "only-correct" and prediction_is_correct is False:
            continue

        if mode == "only-incorrect" and prediction_is_correct is True:
            continue

        with Timer() as timer:
            influences, s_test = influence_helpers.compute_influences_single(
                model=model,
                test_inputs=test_inputs,
                train_dataset=train_dataset,
                batch_size=batch_size,
                data_collator=DataCollatorWithPadding(
                    tokenizer=tokenizer),
                s_test_damp=5e-3,
                s_test_scale=1e4,
                s_test_num_samples=s_test_num_samples,
                s_test_iterations=1 if batch_size == 1 else 4,
                weight_decay=0.005,
                precomputed_grad_train_dict=precomputed_grad_train_dict
            )

            outputs = {
                "test_index": test_index,
                "test_inputs": test_inputs,
                "influences": influences,
                "time": timer.elapsed,
                "correct": prediction_is_correct,
            }
            num_examples_tested += 1
            outputs_collections[test_index] = outputs

            print(f"Status: #{test_index} | {num_examples_tested} / {num_examples_to_test}")
    del precomputed_grad_train_dict
    return outputs_collections


def run_full_influence_functions_dataset(
        model: torch.nn.Module,
        val_dataset: Dataset,
        train_dataset: Dataset,
        tokenizer: AutoTokenizer,
        s_test_num_samples: int = 1000,
        s_test_obj: str = "ce",
        weight_decay: Optional[float] = None
) -> Dict[int, Dict[str, Any]]:
    batch_size = min(max(1, len(train_dataset) // s_test_num_samples), 128)
    print(f"using batch_sise = {batch_size} to calculate s_test")

    with Timer() as timer:
        influences, s_test = influence_helpers.compute_influences_single(
            model=model,
            test_inputs=val_dataset,
            train_dataset=train_dataset,
            batch_size=batch_size,
            data_collator=DataCollatorWithPadding(
                tokenizer=tokenizer),
            s_test_damp=5e-3,
            s_test_scale=1e4,
            s_test_num_samples=s_test_num_samples,
            s_test_iterations=1,
            s_test_obj=s_test_obj,
            weight_decay=weight_decay
        )

        outputs = {
            -1: {"influences": influences,
                 "time": timer.elapsed}
        }
    return outputs


def run_remove_instance(outputs_collections, processor, init_acc, init_loss, train_dataset, val_dataset, test_dataset,
                        save_path):
    helpful_indices_list = []
    harmful_indices_list = []
    n = 0
    for i, output in outputs_collections.items():
        helpful_indices, harmful_indices = misc_utils.get_helpful_harmful_indices_from_influences_dict(
            output['influences'])
        helpful_indices_list.append(helpful_indices)
        harmful_indices_list.append(harmful_indices)
        n += 1
    random_idx = [list(range(len(train_dataset))) for _ in range(n)]
    [np.random.shuffle(_) for _ in random_idx]

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=[10, 5])

    for k, v in [('harmful', harmful_indices_list), ('helpful', helpful_indices_list), ('random', random_idx)]:
        accs = [init_acc]
        xs = [0]
        losses = [init_loss]
        for remove_num in NUMS:
            if remove_num < len(train_dataset) and remove_num // n >= 1:
                remain_idx = set(range(len(train_dataset)))
                for indices in v:
                    remain_idx = remain_idx.difference(set(indices[:remove_num // n]))

                remain_train_dataset = train_dataset.select(remain_idx)
                processor.load_model()
                processor.train(remain_train_dataset, val_dataset)
                val_metric, val_loss = processor.validate(test_dataset)
                accs.append(val_metric['eval_accuracy'])
                losses.append(val_loss)
                xs.append(remove_num)

        axes[0].plot(xs, accs, label=k)
        axes[1].plot(xs, losses, label=k)

    for i, name in enumerate(["Acc.", "Loss"]):
        axes[i].legend()
        axes[i].set_title(f"Remove (Ins.)")
        axes[i].set_xscale("symlog")
        axes[i].set_ylabel(name)
        axes[i].set_xlabel("Number of Removed Instances")
    fig.savefig(save_path)
    plt.close(fig)


def run_avg(outputs_collections, processor, init_acc, init_loss, train_dataset, val_dataset, test_dataset,
            save_path, remove=True):
    sum_influence = np.zeros((len(train_dataset),))
    counter = np.zeros((len(train_dataset),))

    for i, output in outputs_collections.items():
        for k, v in output['influences'].items():
            sum_influence[k] += v
            counter[k] += 1

    avg_influence = np.divide(sum_influence, counter, where=counter != 0, out=np.zeros_like(counter))
    influence_dict = {i: avg_influence[i] for i in range(len(train_dataset))}

    helpful_indices, harmful_indices = misc_utils.get_helpful_harmful_indices_from_influences_dict(influence_dict)
    logging.info(f"#all: {len(train_dataset)}; #helpful: {len(helpful_indices)}; # harmful: {len(harmful_indices)}")

    remain_idx = set(range(len(train_dataset)))
    random_idx = list(range(len(train_dataset)))
    np.random.shuffle(random_idx)

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=[10, 5])

    for k, v in [('harmful', harmful_indices), ('helpful', helpful_indices), ('random', random_idx)]:
        accs = [init_acc]
        xs = [0]
        losses = [init_loss]
        for num in NUMS:
            if num < len(train_dataset):
                if remove:
                    select_idx = remain_idx.difference(set(v[:num]))
                else:
                    select_idx = v[:num]

                processor.load_model()
                processor.train(train_dataset.select(select_idx), val_dataset)
                val_metric, val_loss = processor.validate(test_dataset)
                accs.append(val_metric['eval_accuracy'])
                losses.append(val_loss)
                xs.append(num)

        axes[0].plot(xs, accs, label=k)
        axes[1].plot(xs, losses, label=k)
        print(k, accs)

    for i, name in enumerate(["Acc.", "Loss"]):
        axes[i].legend()
        axes[i].set_title(f"{'Remove' if remove else 'Include'} (Avg.)")
        axes[i].set_xscale("symlog")
        axes[i].set_ylabel(name)
        axes[i].set_xlabel("Number of Instances")
    fig.savefig(save_path)
    plt.close(fig)


def run_avg_percent(outputs_collections, processor, init_acc, init_loss, train_dataset, val_dataset, test_dataset,
                    save_path, class_balance=False, if_corrupted=None):
    sum_influence = np.zeros((len(train_dataset),))
    counter = np.zeros((len(train_dataset),))

    for i, output in outputs_collections.items():
        for k, v in output['influences'].items():
            sum_influence[k] += v
            counter[k] += 1

    avg_influence = np.divide(sum_influence, counter, where=counter != 0, out=np.zeros_like(counter))
    helpful_to_harmful = avg_influence.argsort()

    random_idx = np.arange(0, len(train_dataset))
    np.random.shuffle(random_idx)

    if if_corrupted is not None:
        fig, axes = plt.subplots(1, 3, sharex=True, figsize=[15, 5])
    else:
        fig, axes = plt.subplots(1, 2, sharex=True, figsize=[10, 5])

    # for k, v in [('helpful', helpful_to_harmful), ('harmful', helpful_to_harmful[::-1]), ('random', random_idx)]:
    for k, v in [('helpful', helpful_to_harmful), ('random', random_idx)]:
        accs = [init_acc]
        xs = [0]
        losses = [init_loss]
        if if_corrupted is not None:
            recalls = [0]
        for p in PERCENT:
            if class_balance:
                labels = np.array(train_dataset['label'])[v]
                select_idx = []
                for l in set(labels):
                    if_l = labels == l
                    num = int(p*(if_l.sum()))
                    select_idx.append(v[if_l][num:])
                select_idx = np.concatenate(select_idx)
            else:
                select_idx = v[int(p*len(train_dataset)):]

            if if_corrupted is not None:
                r = 1 - if_corrupted[select_idx].sum() / if_corrupted.sum()
                recalls.append(r)

            processor.load_model()
            processor.train(train_dataset.select(select_idx), val_dataset)
            val_metric, val_loss = processor.validate(test_dataset)
            accs.append(val_metric['eval_accuracy'])
            losses.append(val_loss)
            # accs.append(1)
            # losses.append(1)
            xs.append(p)

        axes[0].plot(xs, accs, label=k)
        axes[1].plot(xs, losses, label=k)
        if if_corrupted is not None:
            axes[2].plot(xs, recalls, label=k)
            print(k, recalls)
        logging.info(f"{k}: {str(accs)}")
        logging.info(f"{k}: {str(losses)}")

    for i, name in enumerate(["Acc.", "Loss"] if if_corrupted is None else ["Acc.", "Loss", "Noise Recall"]):
        axes[i].legend()
        axes[i].set_title(f"Remove (Avg.) based on influence score")
        axes[i].set_ylabel(name)
        axes[i].set_xlabel("Percent of Instances")
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def test_plot():
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=[10, 5])
    for i in range(2):
        axes[0].plot([0, 1, 10, 100], [1] + np.random.random((3,)).tolist(), label=i)
        axes[1].plot([0, 1, 10, 100], [1] + np.random.random((3,)).tolist(), label=i)
    for i, name in enumerate(["Acc.", "Loss"]):
        axes[i].legend()
        axes[i].set_title(f"Remove (Ins.) & Retraining {name}")
        axes[i].set_xscale("symlog")
        axes[i].set_ylabel(name)
        axes[i].set_xlabel("Number of Removed Instances")
    fig.show()
