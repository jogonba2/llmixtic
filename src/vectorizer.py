import gc
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from merge_tokenizers import PythonGreedyCoverageAligner
from merge_tokenizers.types import TokenizedSet
from sklearn.base import BaseEstimator, TransformerMixin
from tokenizers.processors import TemplateProcessing
from torch import Tensor as T
from tqdm import trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from .logging import get_logger
from .quantization import QUANTIZATION_CONFIGS

_logger = get_logger(__name__)


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "bos_token", None) is None:
        tokenizer.bos_token = tokenizer.eos_token
    # Ensure to add BOS and EOS tokens
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
        special_tokens=[
            (tokenizer.eos_token, tokenizer.eos_token_id),
            (tokenizer.bos_token, tokenizer.bos_token_id),
        ],
    )
    return tokenizer


class ProbabilisticVectorizer(BaseEstimator, TransformerMixin):
    """Vectorizer of probabilistic features.
    Uses causal language models to compute 3 kind of features at each timestep.
     - The probability of the observed token.
     - The probability of the most likely token.
     - The entropy.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_names = [
            "observed",
            "most_likely",
            "entropy",
            "median",
            "standard_deviation",
            "top_k",
            "mld",
            "gini",
            "hidden_similarities",
            "hidden_norms",
        ]

    def load_model(self, model_name: str) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **QUANTIZATION_CONFIGS[self.config["quantization"]],
        )
        return model

    def encode(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch: List[str],
        device: str = "cpu",
    ) -> Dict[str, T]:
        encodings = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt",
        )
        # Some tokenizers generate too high word ids
        encodings["input_ids"][
            encodings["input_ids"] >= tokenizer.vocab_size
        ] = 0

        encodings = {k: v.to(device) for k, v in encodings.items()}
        return encodings

    def fit(self, X: List[str]) -> "ProbabilisticVectorizer":
        return self

    def get_number_of_features(self) -> Tuple[int, List[str]]:
        """Calculates the number of features that will be obtained from the vectorizer.

        Since this depends on the specified features, and some features have different
        shapes depending on the model that is used, it must be calculated dynamically.
        """
        n_features = 0
        names = []
        for model_name in self.config["models"]:

            # Features related to hidden states depend on the number
            # of layers of the model, so we load the model and run a single
            # inference step on one example to know these shapes easily
            # This is due to HF not having a canonical name for the layers
            if any("hidden" in feature for feature in self.config["features"]):
                model = self.load_model(model_name)
                tokenizer = load_tokenizer(model_name)
                batch = ["example"]

                encodings = self.encode(tokenizer, batch, model.device)

                with torch.inference_mode():
                    hidden_states = model(
                        input_ids=encodings["input_ids"],
                        attention_mask=encodings["attention_mask"],
                        output_hidden_states=True,
                    ).hidden_states
                    n_layers = len(hidden_states)

            for feature_name in self.config["features"]:
                if feature_name == "hidden_similarities":
                    n_features += n_layers - 1
                    names.extend([feature_name] * (n_layers - 1))
                elif feature_name == "hidden_norms":
                    n_features += n_layers
                    names.extend([feature_name] * n_layers)
                elif feature_name == "top_k":
                    n_features += self.config["top_k"]
                    names.extend([feature_name] * self.config["top_k"])
                else:
                    n_features += 1
                    names.append(feature_name)
        return n_features, names

    def get_feature_cache_path(
        self, cache_name: str, model_name: str, feature_name: str
    ) -> Path:
        """
        Constructs the path to the cache for a model and name
        """
        assert feature_name in self.feature_names

        base_path = Path(self.config["cache_dir"]) / cache_name
        save_name = "_".join(model_name.split("/"))

        cache_path = base_path / save_name
        cache_path.mkdir(parents=True, exist_ok=True)

        feature_path = cache_path / f"{feature_name}.pt"
        return feature_path

    def load_feature(
        self, cache_name: str, model_name: str, feature_name: str
    ) -> T:
        """Loads a single feature"""
        feature_path = self.get_feature_cache_path(
            cache_name, model_name, feature_name
        )
        return torch.load(feature_path)

    def save_feature(
        self, feature: T, cache_name: str, model_name: str, feature_name: str
    ) -> None:
        """Saves a single feature"""
        feature_path = self.get_feature_cache_path(
            cache_name, model_name, feature_name
        )
        return torch.save(feature, feature_path)

    def load_features(
        self, cache_name: str, model_name: str
    ) -> Union[Dict[str, T], None]:
        """Loads all specified features"""
        try:
            features = {}
            loaded_features = []
            for feature_name in self.feature_names:
                if feature_name in self.config["features"]:
                    features[feature_name] = self.load_feature(
                        cache_name, model_name, feature_name
                    ).cpu()
                    loaded_features.append(feature_name)
            _logger.info(
                f"*************LOADED {model_name} - {cache_name} features: {', '.join(loaded_features)} ************"
            )
            return features
        except FileNotFoundError:
            return None

    def transform(self, X: List[str], cache_name: str) -> T:
        """Transform method to get feature vectors based on probs from LMs
        If an LLM is employed for the first time with a given cache name and dataset,
        all features will be generated and saved in the cache path provided in the config.
        Otherwise, it will only load the features that have been specified.

        Args:
            X (List[str]): list of N texts
            cache_name (str): the name of the cache dir to be used
        Returns:
            Tensor: vectors of k features for each text in X, with shape
                (N, max_length-1, k) (BOS excluded)
        """
        model_features: Dict[str, Dict[str, T]] = {
            model_name: {feature: T([]) for feature in self.feature_names}
            for model_name in self.config["models"]
        }

        # Prepare model tokens when `merge_tokens` is True
        if self.config.get("merge_tokens", False):
            model_tokens = {
                model_name: [] for model_name in self.config["models"]
            }
            for model_name in self.config["models"]:
                tokenizer = load_tokenizer(model_name)
                model_tokens[model_name] = [
                    tokenizer.convert_ids_to_tokens(ids)
                    for ids in self.encode(tokenizer, X, "cpu")["input_ids"]
                ]

        for model_name in self.config["models"]:
            loaded_features = self.load_features(cache_name, model_name)
            if loaded_features:
                model_features[model_name] = loaded_features
                continue

            model = self.load_model(model_name)
            tokenizer = load_tokenizer(model_name)

            for sample_idx in trange(
                0,
                len(X),
                self.config["batch_size"],
                desc=f"Featurizing with {model_name}",
            ):
                batch = X[sample_idx : sample_idx + self.config["batch_size"]]

                # Warning: we assume all tokenizers produce the same number of tokens
                encodings = self.encode(tokenizer, batch, model.device)
                target_ids = encodings["input_ids"].clone()

                with torch.inference_mode():
                    # Compute logits
                    outputs = model(
                        input_ids=encodings["input_ids"],
                        attention_mask=encodings["attention_mask"],
                        output_hidden_states=True,
                    )
                    logits, hidden_states = (
                        outputs.logits,
                        outputs.hidden_states,
                    )

                    # Shift logits, targets, mask and hidden states
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_targets = target_ids[..., 1:].contiguous()
                    mask = encodings["attention_mask"][..., 1:].contiguous()
                    shift_hidden_states = [
                        hidden_state[..., :-1, :].contiguous()
                        for hidden_state in hidden_states
                    ]

                    # Get probabilities
                    probs = shift_logits.softmax(dim=-1)
                    smallest_normal = torch.finfo(
                        type=probs.dtype
                    ).smallest_normal
                    probs[probs == 0] = smallest_normal

                    batch_features = self.compute_features(
                        probs,
                        shift_targets,
                        mask,
                        shift_hidden_states,
                        eps=smallest_normal,
                    )

                    for feature_name, feature in batch_features.items():
                        model_features[model_name][feature_name] = torch.cat(
                            (
                                model_features[model_name][feature_name],
                                feature.cpu(),
                            ),
                            dim=0,
                        )

            # Cache current model features
            for feature_name, feature in model_features[model_name].items():
                self.save_feature(feature, cache_name, model_name, feature_name)

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # In cases where a model doesnt have a previously cached feature, every
        # feature is calculated and cached, but we must delete the ones that
        # were not specified in the config for the specific experiment
        for model_name in self.config["models"]:
            for feature_name in self.feature_names:
                if (
                    feature_name not in self.config["features"]
                    and feature_name in model_features[model_name]
                ):
                    del model_features[model_name][feature_name]

            model_features[model_name] = torch.cat(
                list(model_features[model_name].values()), dim=-1
            ).cpu()

            _logger.info(
                f"Per model features for model {model_name}: {model_features[model_name].shape[-1]}"  # type: ignore
            )

        # Merge features with merge-tokenizers
        if self.config.get("merge_tokens", False):
            aligner = PythonGreedyCoverageAligner()
            features = T([])
            for i in trange(
                0, len(X), desc="Aligning tokens with merge-tokenizers..."
            ):
                sample_tokens = [
                    model_tokens[model_name][i][1:]
                    for model_name in model_tokens
                ]
                sample_features = [
                    model_features[model_name][i].cpu().numpy()
                    for model_name in model_tokens
                ]
                aggregated_features = aligner.aggregate_features(
                    TokenizedSet(
                        tokens=sample_tokens,
                        features=sample_features,
                        text=X[i],
                    ),
                    stack=True,
                )
                features = torch.cat(
                    (
                        features,
                        torch.from_numpy(aggregated_features)
                        .unsqueeze(0)
                        .float(),
                    )
                )
        else:
            features = torch.cat(list(model_features.values()), dim=-1)

        features = features.cpu()
        _logger.info(f"Total features: {features.shape[-1]}")
        return features

    def get_observed(
        self, probs: T, shift_targets: T, mask: T, eps: float = 1e-14
    ) -> T:
        """Computes the log probabilities of the target tokens

        Args:
            probs (Tensor, float32): probabilities across all the timesteps
                except the last one (EOS token as input), with shape
                (batch_size, max_length-1, vocab_size)
            shift_targets (Tensor, int64): token ids shifted one step to
                the right (targets), with shape (batch_size, max_length-1)
            mask (Tensor, int64): attention mask shifted one step to the
                right, with shape (batch_size, max_length-1)

        Returns:
            Tensor (float32): log probabilities of the target tokens, except
                for the last one (EOS token as input),
                with shape (batch_size, max_length-1)
        """
        observed = torch.log(
            torch.gather(
                probs, dim=-1, index=shift_targets.unsqueeze(dim=-1)
            ).squeeze(dim=-1)
            + eps
        )
        observed = observed * mask
        return observed

    def get_most_likely(self, probs: T, mask: T, eps: float = 1e-14) -> T:
        """Computes the log probability of the most likely
        token (according to the model)

        Args:
            probs (Tensor, float32): probabilities across all the timesteps
                except the last one (EOS token as input), with shape
                (batch_size, max_length-1, vocab_size)
            mask (Tensor, int64): attention mask shifted one step to the
                right, with shape (batch_size, max_length-1)
        Returns:
            Tensor (float32): log probabilities of the most likely tokens,
                except for the last one (EOS token as input),
                with shape (batch_size, max_length-1)
        """
        most_likely = torch.log(torch.max(probs, dim=-1).values + eps)
        most_likely = most_likely * mask
        return most_likely

    def get_entropy(self, probs: T, mask: T, eps: float = 1e-14) -> T:
        """Computes the entropy of the distribution at each timestep

        Args:
            probs (Tensor, float32): probabilities across all the timesteps
                except the last one (EOS token as input), with shape
                (batch_size, max_length-1, vocab_size)
            mask (Tensor, int64): attention mask shifted one step to the
                right, with shape (batch_size, max_length-1)
            eps (float): epsilon to avoid log2(0)

        Returns:
            Tensor (float32): entropy at each timestep, except for the last one
                (EOS token as input), with shape (batch_size, max_length-1)
        """
        entropy = -torch.sum(probs * torch.log2(probs + eps), dim=-1)
        entropy = entropy * mask
        return entropy

    def get_median(self, probs: T, mask: T, eps: float = 1e-14) -> T:
        """Computes the log median of the distribution at each timestep

        Args:
            probs (Tensor, float32): probabilities across all the timesteps
                except the last one (EOS token as input), with shape
                (batch_size, max_length-1, vocab_size)
            mask (Tensor, int64): attention mask shifted one step to the
                right, with shape (batch_size, max_length-1)

        Returns:
            Tensor (float32): median at each timestep, except for the last one
                (EOS token as input), with shape (batch_size, max_length-1)
        """
        median = torch.log(probs.median(dim=-1).values + eps)
        median = median * mask
        return median

    def get_standard_deviation(
        self, probs: T, mask: T, eps: float = 1e-14
    ) -> T:
        """Computes the log st.dev. of the distribution at each timestep

        Args:
            probs (Tensor, float32): probabilities across all the timesteps
                except the last one (EOS token as input), with shape
                (batch_size, max_length-1, vocab_size)
            mask (Tensor, int64): attention mask shifted one step to the
                right, with shape (batch_size, max_length-1)

        Returns:
            Tensor (float32): st.dev at each timestep, except for the last one
                (EOS token as input), with shape (batch_size, max_length-1)
        """
        stdev = torch.log(probs.std(dim=-1) + eps)
        stdev = stdev * mask
        return stdev

    def get_top_k(self, probs: T, mask: T, eps: float = 1e-14) -> T:
        """Computes the top k most probable tokens at each timestep
        Given that top-1 is just the most_likely feature, we instead return
        the top-10 features from 2nd to 11th, ignoring most_likely.

        Args:
            probs (Tensor, float32): probabilities across all the timesteps
                except the last one (EOS token as input), with shape
                (batch_size, max_length-1, vocab_size)
            mask (Tensor, int64): attention mask shifted one step to the
                right, with shape (batch_size, max_length-1)

        Returns:
            Tensor (float32): 2nd to (k+1)-th most probable tokens at each timestep,
                 except for the last one (EOS token as input),
                 with shape (batch_size, max_length-1, k)
        """
        k = self.config.get("top_k", 10)
        top_k = probs.topk(k + 1, dim=-1).values[..., 1:]
        top_k = torch.log(top_k + eps)
        top_k = top_k * mask.unsqueeze(dim=-1)
        return top_k

    def get_mld(self, probs: T, mask: T, eps: float = 1e-14) -> T:
        """Computes the mean log deviation at each timestep, defined as
        the difference between the log of the mean and the mean of the logs of the
        values in a given distribution, i.e.:

        MLD = log(mean(x)) - mean(log(x))
        https://en.wikipedia.org/wiki/Mean_log_deviation

        Args:
            probs (Tensor, float32): probabilities across all the timesteps
                except the last one (EOS token as input), with shape
                (batch_size, max_length-1, vocab_size)
            mask (Tensor, int64): attention mask shifted one step to the
                right, with shape (batch_size, max_length-1)

        Returns:
            Tensor (float32): sum of top k most probable tokens at each timestep,
                 except for the last one (EOS token as input),
                 with shape (batch_size, max_length-1)
        """
        log_mean = torch.log(probs.mean(dim=-1) + eps)
        mean_logs = torch.log(probs + eps).mean(dim=-1)
        mld = log_mean - mean_logs
        mld = mld * mask

        return mld

    def get_gini(self, probs: T, mask: T, eps: float = 1e-14) -> T:
        """Computes the log gini impurity at each timestep.

        https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity

        Args:
            probs (Tensor, float32): probabilities across all the timesteps
                except the last one (EOS token as input), with shape
                (batch_size, max_length-1, vocab_size)
            mask (Tensor, int64): attention mask shifted one step to the
                right, with shape (batch_size, max_length-1)

        Returns:
            Tensor (float32): log gini impurity at each timestep,
                 except for the last one (EOS token as input),
                 with shape (batch_size, max_length-1)
        """
        gini = torch.log(1 - probs.square().sum(dim=-1) + eps)
        gini = gini * mask
        return gini

    def get_hidden_state_similarities(
        self, shift_hidden_states: List[T], mask: T
    ) -> T:
        """Computes pairwise hidden state cosine similarities at each timestep.
        Specifically, given a sequence of hidden layers
        H = (h_0, h_1, ..., h_n)
        we return [sim(h_0, h_1), sim(h_1, h_2), ..., sim(h_{n-1}, h_n)]

        Args:
            shift_hidden_states (List[Tensor], float32): hidden states for all timesteps
                except the last one (EOS token as input), each with shape
                (batch_size, max_length-1, hidden_dim)
            mask (Tensor, int64): attention mask shifted one step to the
                right, with shape (batch_size, max_length-1)

        Returns:
            Tensor (float32): h-1 hidden state similarities at each timestep,
                 except for the last one (EOS token as input),
                 with shape (batch_size, max_length-1, h-1)
        """
        similarities = [
            F.cosine_similarity(h1, h2, dim=-1) * mask
            for h1, h2 in zip(shift_hidden_states[:-1], shift_hidden_states[1:])
        ]
        similarities = torch.stack(similarities, dim=-1)
        return similarities

    def get_hidden_state_norms(
        self, shift_hidden_states: List[T], mask: T
    ) -> T:
        """Computes hidden state 2-norms at each timestep.

        Args:
            shift_hidden_states (List[Tensor], float32): hidden states for all timesteps
                except the last one (EOS token as input), each with shape
                (batch_size, max_length-1, hidden_dim)
            mask (Tensor, int64): attention mask shifted one step to the
                right, with shape (batch_size, max_length-1)

        Returns:
            Tensor (float32): h hidden state norms at each timestep,
                 except for the last one (EOS token as input),
                 with shape (batch_size, max_length-1, h)
        """
        norms = [
            torch.linalg.vector_norm(h, dim=-1) * mask
            for h in shift_hidden_states
        ]
        norms = torch.stack(norms, dim=-1)
        return norms

    def compute_features(
        self,
        probs: T,
        shift_targets: T,
        mask: T,
        shift_hidden_states: List[T],
        eps: float = 1e-14,
    ) -> Dict:
        """Computes the probabilistic features

        Args:
            probs (Tensor, float32): probabilities across all the timesteps
                except the last one (EOS token as input), with shape
                (batch_size, max_length-1, vocab_size)
            shift_targets (Tensor, int64): token ids shifted one step to
                the right (targets), with shape (batch_size, max_length-1)
            mask (Tensor, int64): attention mask shifted one step to the
                right, with shape (batch_size, max_length-1)
            shift_hidden_states (List[Tensor], float32): hidden states for all timesteps
                except the last one (EOS token as input), each with shape
                (batch_size, max_length-1, hidden_dim)

        Returns:
            Dict[str, Tensor] (float32): the k probabilistic features per-token, except
                for the last one (EOS token as input)
                with shape (batch_size, max_length-1, feature_dim)
        """

        # Feature 1: Log probability of the observed token
        observed = self.get_observed(probs, shift_targets, mask, eps)

        # Feature 2: Log probability of the most likely token (according to the model)
        most_likely = self.get_most_likely(probs, mask, eps)

        # Feature 3: Entropy of the distribution at each position
        entropy = self.get_entropy(probs, mask, eps)

        # Feature 4: Log median of the distribution at each position
        median = self.get_median(probs, mask, eps)

        # Feature 5: standard deviation of the distribution at each position
        standard_deviation = self.get_standard_deviation(probs, mask, eps)

        # Feature 6: Sum of log-probabilities of k most probable tokens
        top_k = self.get_top_k(probs, mask, eps)

        # Feature 7: Mean log deviation of the distribution at each position
        mld = self.get_mld(probs, mask, eps)

        # Feature 8: log gini impurity
        gini = self.get_gini(probs, mask, eps)

        # Features 9: cosine similarities of hidden states
        hidden_similarities = self.get_hidden_state_similarities(
            shift_hidden_states, mask
        )

        # Features 10: norms of hidden states
        hidden_norms = self.get_hidden_state_norms(shift_hidden_states, mask)

        features = {
            "observed": observed.unsqueeze(dim=-1),
            "most_likely": most_likely.unsqueeze(dim=-1),
            "entropy": entropy.unsqueeze(dim=-1),
            "median": median.unsqueeze(dim=-1),
            "standard_deviation": standard_deviation.unsqueeze(dim=-1),
            "top_k": top_k,
            "mld": mld.unsqueeze(dim=-1),
            "gini": gini.unsqueeze(dim=-1),
            "hidden_similarities": hidden_similarities,
            "hidden_norms": hidden_norms,
        }

        return features
