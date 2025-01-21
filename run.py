import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from _jsonnet import evaluate_file
from datasets import concatenate_datasets, load_from_disk
from sklearn.metrics import accuracy_score, f1_score

from src.llmixtic import LLMixtic


def set_seed(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_classification(
    refs: List[int], preds: List[int], model: LLMixtic
) -> Dict:
    return {
        "features": ", ".join(model.feature_params["features"]),
        "models": ", ".join(model.feature_params["models"]),
        "accuracy": accuracy_score(refs, preds),
        "macro-f1": f1_score(refs, preds, average="macro"),
        "micro-f1": f1_score(refs, preds, average="micro"),
    }


def parse_config(
    path: Path,
) -> Dict:
    config = json.loads(evaluate_file(str(path)))
    assert all(x in config for x in {"run_name", "train", "test", "model"})

    return config


def run_single(
    train_datasets: Dict[str, Dict[str, str]],
    test_datasets: Dict[str, Dict[str, str]],
    model_params: Dict,
    save_dir: Path,
) -> None:
    set_seed(0)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = LLMixtic(**model_params)

    # All train datasets are concatenated
    loaded_train_datasets, train_names = [], []
    for name, params in train_datasets.items():
        dataset = load_from_disk(params["dataset_path"])[params["split"]]
        loaded_train_datasets.append(dataset)
        train_names.append(name)

    train_data = concatenate_datasets(loaded_train_datasets)
    train_name = "_and_".join(train_names)
    train_data = train_data.select(range(10)) # OJO
    model.fit(train_data, train_name)

    # The test datasets are considered one-by-one
    all_results = {}
    for name, params in test_datasets.items():
        dataset = load_from_disk(params["dataset_path"])[params["split"]]
        dataset = dataset.select(range(10)) # OJO
        preds = model.predict(dataset, name)

        results = evaluate_classification(dataset["label"], preds, model)
        all_results[name] = results

        df = pd.DataFrame({name: results}).T.reset_index()
        df.columns = ["dataset"] + list(results.keys())
        df.to_csv(save_dir / f"{name}.tsv", sep="\t", index=False)
        df.to_markdown(save_dir / f"{name}.md", index=False)

    df = pd.DataFrame(all_results).T.reset_index()
    df.columns = ["dataset"] + list(results.keys())
    df.to_csv(save_dir / "all_test.tsv", sep="\t", index=False)
    df.to_markdown(save_dir / "all_test.md", index=False)


def run_importance(
    train_datasets: Dict[str, Dict[str, str]],
    test_datasets: Dict[str, Dict[str, str]],
    model_params: Dict,
    save_dir: Path,
) -> None:
    set_seed(0)
    if save_dir.exists():
        save_dir = save_dir / f"{save_dir.name}_importance"

    save_dir.mkdir(parents=True, exist_ok=True)
    model = LLMixtic(**model_params)

    assert (
        len(test_datasets) == 1
    ), "It is recommended to only run importance on one dataset"

    name, params = next(iter(test_datasets.items()))
    dataset = load_from_disk(params["dataset_path"])[params["split"]]
    importances = model.permutation_feature_importance(dataset, name)

    for name, df in importances.items():
        df.to_csv(save_dir / f"{name}.tsv", sep="\t", index=False)
        df.to_markdown(save_dir / f"{name}.md", index=False)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--importance", action="store_true")

    args = parser.parse_args()

    config = parse_config(args.config)
    save_dir = args.save_dir / config["run_name"]

    assert (
        args.train != args.importance
    ), "Only one of --train and --importance can be chosen"
    run = run_single if args.train else run_importance

    run(config["train"], config["test"], config["model"], save_dir)
