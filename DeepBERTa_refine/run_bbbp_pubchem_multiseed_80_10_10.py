#!/usr/bin/env python3
"""
Fine-tune and evaluate pretrained DeepBERTa/RoBERTa models on BBBP using
fixed scaffold-based 80/10/10 refine/validation/benchmark splits.

Expected CSV columns:
    sequence,deepsmiles,p_np

Example:
    python run_bbbp_pubchem_multiseed_80_10_10.py \
      --model_name_or_path /path/to/base/model \
      --train_csv bbbp_scaffold_refine80.csv \
      --val_csv bbbp_scaffold_val10.csv \
      --bench_csv bbbp_scaffold_bench10.csv \
      --output_root_dir ./bbbp_results \
      --seeds 1,2,3
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

BBBP_TASK = "p_np"


def parse_label(value):
    """Return 0/1 for valid labels and -100 for missing/invalid labels."""
    if value is None:
        return -100

    value = str(value).strip()
    if value == "" or value.upper() in {"NA", "NAN", "NONE", "NULL"}:
        return -100

    try:
        numeric = float(value)
    except ValueError:
        return -100

    if numeric == 0.0:
        return 0
    if numeric == 1.0:
        return 1
    return -100


def load_bbbp_csv(csv_path):
    texts = []
    labels = []

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []

        if "deepsmiles" not in fieldnames:
            raise ValueError(
                f"'deepsmiles' column not found in {csv_path}. "
                f"Found columns: {fieldnames}"
            )

        if BBBP_TASK not in fieldnames:
            raise ValueError(
                f"'{BBBP_TASK}' label column not found in {csv_path}. "
                f"Found columns: {fieldnames}"
            )

        for row in reader:
            deepsmiles = (row.get("deepsmiles") or "").strip()
            if not deepsmiles:
                continue

            texts.append(deepsmiles)
            labels.append([parse_label(row.get(BBBP_TASK))])

    if not texts:
        raise ValueError(f"No usable rows found in {csv_path}")

    return Dataset.from_dict(
        {
            "deepsmiles": texts,
            "labels": labels,
        }
    )


class BBBPTrainer(Trainer):
    """Binary BCEWithLogitsLoss with masking of missing labels (-100)."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        valid_mask = labels != -100

        safe_labels = labels.clone()
        safe_labels[~valid_mask] = 0

        loss_matrix = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            safe_labels.float(),
            reduction="none",
        )
        loss_matrix = loss_matrix * valid_mask.float()

        denominator = valid_mask.float().sum()
        loss = (
            loss_matrix.sum() / denominator
            if denominator.item() > 0
            else loss_matrix.mean()
        )

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    y_true = labels[:, 0]
    valid_mask = y_true != -100

    y_true = y_true[valid_mask]
    y_score = logits[valid_mask, 0]

    if y_true.size < 2 or np.unique(y_true).size < 2:
        return {
            "roc_auc_p_np": 0.0,
            "mean_roc_auc": 0.0,
        }

    auc = float(roc_auc_score(y_true, y_score))

    return {
        "roc_auc_p_np": auc,
        "mean_roc_auc": auc,
    }


def print_label_distribution(dataset, split_name):
    labels = np.asarray(dataset["labels"])[:, 0]
    print(
        f"{split_name} p_np distribution: "
        f"0={int(np.sum(labels == 0))}, "
        f"1={int(np.sum(labels == 1))}, "
        f"missing={int(np.sum(labels == -100))}"
    )


def run_one_seed(seed, args, train_dataset, val_dataset, bench_dataset):
    set_seed(seed)

    tokenizer = RobertaTokenizerFast.from_pretrained(
        args.model_name_or_path
    )

    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    run_dir = os.path.join(args.output_root_dir, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(run_dir, "logs"),
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="mean_roc_auc",
        greater_is_better=True,
        seed=seed,
        data_seed=seed,
        report_to=args.report_to,
    )

    trainer = BBBPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_metrics = trainer.evaluate(
        val_dataset,
        metric_key_prefix="val",
    )
    bench_metrics = trainer.evaluate(
        bench_dataset,
        metric_key_prefix="bench",
    )

    result = {
        "seed": seed,
        "validation": val_metrics,
        "benchmark": bench_metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_metric": trainer.state.best_metric,
    }

    with open(
        os.path.join(run_dir, "bbbp_results.json"),
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(result, output_file, indent=2)

    with open(
        os.path.join(run_dir, "bbbp_results_summary.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["seed", seed])
        writer.writerow([])
        writer.writerow(
            ["val_roc_auc_p_np", val_metrics.get("val_roc_auc_p_np")]
        )
        writer.writerow(
            ["bench_roc_auc_p_np", bench_metrics.get("bench_roc_auc_p_np")]
        )

    trainer.save_model(os.path.join(run_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(run_dir, "best_model"))

    val_auc = float(val_metrics.get("val_roc_auc_p_np", 0.0))
    bench_auc = float(bench_metrics.get("bench_roc_auc_p_np", 0.0))

    print(
        f"[Seed {seed}] "
        f"val_roc_auc_p_np={val_auc:.4f} "
        f"bench_roc_auc_p_np={bench_auc:.4f}"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed BBBP scaffold 80/10/10 evaluation."
    )
    parser.add_argument(
        "--model_name_or_path",
        required=True,
    )
    parser.add_argument(
        "--train_csv",
        default="bbbp_scaffold_refine80.csv",
    )
    parser.add_argument(
        "--val_csv",
        default="bbbp_scaffold_val10.csv",
    )
    parser.add_argument(
        "--bench_csv",
        default="bbbp_scaffold_bench10.csv",
    )
    parser.add_argument(
        "--output_root_dir",
        default="./bbbp_pubchem_80_10_10",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument(
        "--seeds",
        default="1,2,3",
        help="Comma-separated training seeds.",
    )
    parser.add_argument(
        "--report_to",
        default="none",
    )
    args = parser.parse_args()

    seeds = [
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    ]
    if not seeds:
        raise ValueError("At least one training seed is required.")

    train_raw = load_bbbp_csv(args.train_csv)
    val_raw = load_bbbp_csv(args.val_csv)
    bench_raw = load_bbbp_csv(args.bench_csv)

    print(
        f"Rows: train={len(train_raw)}, "
        f"validation={len(val_raw)}, "
        f"benchmark={len(bench_raw)}"
    )
    print_label_distribution(train_raw, "Refine/train")
    print_label_distribution(val_raw, "Validation")
    print_label_distribution(bench_raw, "Benchmark")

    tokenizer = RobertaTokenizerFast.from_pretrained(
        args.model_name_or_path
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["deepsmiles"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = train_raw.map(tokenize_batch, batched=True)
    val_dataset = val_raw.map(tokenize_batch, batched=True)
    bench_dataset = bench_raw.map(tokenize_batch, batched=True)

    tensor_columns = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=tensor_columns)
    val_dataset.set_format(type="torch", columns=tensor_columns)
    bench_dataset.set_format(type="torch", columns=tensor_columns)

    os.makedirs(args.output_root_dir, exist_ok=True)

    results = []
    for seed in seeds:
        print(f"\n========== BBBP seed {seed} ==========")
        results.append(
            run_one_seed(
                seed,
                args,
                train_dataset,
                val_dataset,
                bench_dataset,
            )
        )

    benchmark_values = np.asarray(
        [
            result["benchmark"].get("bench_roc_auc_p_np", 0.0)
            for result in results
        ],
        dtype=float,
    )

    average = float(benchmark_values.mean())
    value_range = float(
        benchmark_values.max() - benchmark_values.min()
    )

    summary_path = os.path.join(
        args.output_root_dir,
        "bbbp_80_10_10_multiseed_summary.csv",
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ["seed", "val_roc_auc_p_np", "bench_roc_auc_p_np"]
        )

        for result in results:
            writer.writerow(
                [
                    result["seed"],
                    result["validation"].get("val_roc_auc_p_np"),
                    result["benchmark"].get("bench_roc_auc_p_np"),
                ]
            )

        writer.writerow([])
        writer.writerow(["benchmark_average_roc_auc", average])
        writer.writerow(["benchmark_range_roc_auc", value_range])

    print("\n===== BBBP Benchmark Summary =====")
    for result in results:
        value = float(
            result["benchmark"].get("bench_roc_auc_p_np", 0.0)
        )
        print(
            f"Seed {result['seed']}: "
            f"bench_roc_auc_p_np={value:.4f}"
        )

    print(
        f"Benchmark Average ± Range: "
        f"{average:.4f} ± {value_range:.4f}"
    )
    print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()
