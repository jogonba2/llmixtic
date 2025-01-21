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
            "meta-llama/Llama-2-7b-chat-hf",
            "Qwen/Qwen2-1.5B-Instruct"
        ],
        "quantization": "int4_bf16",
        "batch_size": 32, 
        "max_length": 512,
        "top_k": 10,
        "features": [
            "observed",
            "most_likely",
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
