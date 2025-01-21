from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import safetensors
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from torch import Tensor as T
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from tqdm import trange
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from .vectorizer import ProbabilisticVectorizer


class LLMixticTransformerEncoder(nn.Module):
    def __init__(
        self,
        model_params: Dict,
        feature_params: Dict,
        n_features: int,
    ):
        super().__init__()
        self.model_params = model_params
        self.feature_params = feature_params
        self.n_features = n_features

        self.layer_norm = nn.LayerNorm(n_features)
        self.inp_proj_layer = nn.Linear(n_features, model_params["d_model"])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_params["d_model"],
            nhead=model_params["n_head"],
            dim_feedforward=model_params["dim_feedforward"],
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=model_params["n_layers"]
        )
        self.linear_layer = nn.Linear(
            model_params["d_model"], model_params["num_labels"]
        )

    def forward(
        self,
        input_ids: T,
        attention_mask: Optional[T] = None,
        labels: Optional[T] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        normalized_input = self.layer_norm(input_ids)
        inp_proj = self.inp_proj_layer(normalized_input)
        hidden_state = self.encoder(inp_proj)
        pooled = hidden_state.mean(axis=1)
        logits = self.linear_layer(pooled)

        # Compute loss
        loss = self.compute_loss(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def compute_loss(
        self, logits: T, labels: Optional[T] = None
    ) -> Optional[T]:
        """Method to compute the loss.

        Args:
            logits (torch.Tensor): logits of the model (last model output)
            labels (torch.Tensor): labels
        Returns:
            torch.Tensor: the loss.
        """
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.model_params["num_labels"]),
                labels.view(-1),
            )
        return loss


class LLMixticDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: ProbabilisticVectorizer,
        cache_name: str,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cache_name = cache_name
        self.input_ids = self.tokenizer.transform(
            dataset["text"], cache_name=self.cache_name
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        label = self.dataset[idx]["label"]
        input_ids = self.input_ids[idx, ...]
        # input_ids = input_ids.nan_to_num()

        return {
            "label": label,
            "input_ids": input_ids,
        }


class PermutationFeatureImportanceWrapper:
    def __init__(self, dataset: LLMixticDataset) -> None:
        self.dataset = dataset

    def run(
        self, model: "LLMixtic", n_repeat: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Runs permutation feature importance.

        For each feature, shuffles the dimension and predicts n_repeat times.

        """
        scores: Dict[int, List[float]] = {}

        n_features, names = model.tokenizer.get_number_of_features()

        for feature in trange(n_features, desc="Feature importance"):
            for _ in range(n_repeat):
                feature_vector = self.dataset.input_ids[..., feature]
                idx = torch.randperm(feature_vector.nelement())
                # unroll, shuffle and reshape to shuffle across batch and seqlen
                feature_vector = feature_vector.view(-1)[idx].view(
                    feature_vector.size()
                )

                self.dataset.input_ids[..., feature] = feature_vector
                preds = model.predict_tokenized(self.dataset)
                score = f1_score(self.dataset.dataset["label"], preds)
                scores[feature] = scores.get(feature, []) + [score]

                # return to original ordering
                self.dataset.input_ids[..., feature] = feature_vector.view(-1)[
                    idx.argsort()
                ].view(feature_vector.size())

        importances = {k: [np.mean(v), np.std(v)] for k, v in scores.items()}
        importances_df = pd.DataFrame(importances).T.reset_index()
        importances_df.columns = ["feature", "mean_macro-f1", "std"]
        importances_df.insert(0, "feature_name", names)
        importances_df = importances_df.sort_values(
            by="mean_macro-f1", ascending=False
        )

        all_runs_df = pd.DataFrame(scores).T.reset_index()
        all_runs_df.columns = ["feature"] + [
            f"run_{i}_macro-f1" for i in range(n_repeat)
        ]
        all_runs_df.insert(0, "feature_name", names)

        return {
            "importances": importances_df,
            "all_importance_runs": all_runs_df,
        }


class LLMixtic:
    def __init__(
        self,
        model_params: Dict,
        feature_params: Dict,
        training_params: Dict,
        inference_params: Dict,
    ):
        self.model_params = model_params
        self.feature_params = feature_params
        self.training_params = training_params
        self.inference_params = inference_params

        self.tokenizer = ProbabilisticVectorizer(self.feature_params)
        n_features, names = self.tokenizer.get_number_of_features()

        self.model = LLMixticTransformerEncoder(
            model_params, feature_params, n_features
        )

        # load checkpoint if given
        if "pretrained_model_name_or_path" in model_params:
            safetensors.torch.load_model(
                self.model, model_params["pretrained_model_name_or_path"]
            )

    def permutation_feature_importance(
        self, dataset: Dataset, cache_name: str
    ) -> Dict[str, pd.DataFrame]:
        tok_dataset = LLMixticDataset(
            dataset, self.tokenizer, cache_name=cache_name
        )

        permutation_importance = PermutationFeatureImportanceWrapper(
            tok_dataset
        )

        return permutation_importance.run(self)

    def predict(self, dataset: Dataset, cache_name: str) -> List[int]:
        tok_dataset = LLMixticDataset(
            dataset, self.tokenizer, cache_name=cache_name
        )
        return self.predict_tokenized(tok_dataset)

    def predict_tokenized(self, tok_dataset: LLMixticDataset) -> List[int]:
        inference_args = TrainingArguments(
            do_predict=True, **self.inference_params
        )
        trainer = Trainer(
            model=self.model,
            args=inference_args,
        )
        preds = trainer.predict(test_dataset=tok_dataset).predictions
        pred_labels = preds.argmax(axis=-1)
        return pred_labels

    def fit(self, dataset: Dataset, cache_name: str) -> None:
        tok_dataset = LLMixticDataset(dataset, self.tokenizer, cache_name)
        return self.fit_tokenized(tok_dataset)

    def fit_tokenized(self, tok_dataset: LLMixticDataset) -> None:
        training_args = TrainingArguments(do_train=True, **self.training_params)

        trainer = Trainer(
            model=self.model,
            train_dataset=tok_dataset,
            args=training_args,
        )
        trainer.train()
