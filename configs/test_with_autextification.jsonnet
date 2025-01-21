local language = "en";
local task = "detection"; # One of "detection", "attribution", "family_attribution"
local run_name = "test_with_autextification";

local train_data = {
    "autextification2023_train": {
        dataset_path: std.format("az://snlp-data/genaios_mgt_detection_attribution/%s/autextification2023/%s/", [language, task]),
        split: "train",
    },
};

local test_data = {
    "autextification2023_test": train_data.autextification2023_train + {split: "test"},
};

local model = {
    "feature_params": {
        "models": [
            "gpt2",
            "distilbert/distilgpt2",
            "gpt2-medium",
            #"meta-llama/Llama-2-13b-chat-hf",
            #"meta-llama/Llama-2-13b-hf",
            #"meta-llama/Llama-2-7b-hf",
            #"meta-llama/Llama-2-7b-chat-hf"
        ],
        "quantization": "int4_bf16",
        "batch_size": 32, 
        "max_length": 512,
        "top_k": 10,
        "features": [
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
        ],
        "cache_dir": std.format("./cache/%s", run_name),
        "merge_tokens": false
    },
    "model_params": {
        "num_labels": 2,
        "d_model": 128,
        "n_head": 4,
        "dim_feedforward": 64,
        "n_layers": 1
    },
    "training_params": {
        "output_dir": std.format("./checkpoints/%s/training", run_name),
        "save_total_limit": 1,
        "per_device_train_batch_size": 16,
        "num_train_epochs": 10,
        "fp16": true,
        "logging_steps": 500,
        "learning_rate": 0.001,
        #"load_best_model_at_end": true,
        #"save_strategy": "steps",
        #"evaluation_strategy": "steps",
        #"eval_steps": 250,
        #"do_eval": true
    },
    "inference_params": {
        "per_device_eval_batch_size": 32,
        "output_dir": std.format("./checkpoints/%s/inference", run_name),
    }
};

{
    "run_name": run_name,
    "train": train_data,
    "test": test_data,
    "model": model,
}
