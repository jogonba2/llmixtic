local language = "en";
local task = "detection"; # One of "detection", "attribution", "family_attribution"
local run_name = "genaios_mgt_detection";

local train_data = {
    "gpt4instructions_train": {
        dataset_path: std.format("az://snlp-data/genaios_mgt_detection_attribution/%s/gpt4-alpaca-instructions/%s/", [language, task]),
        split: "train",
    }, 
    "deepfake_train": {
        dataset_path: std.format("az://snlp-data/genaios_mgt_detection_attribution/%s/deepfake/%s/", [language, task]),
        split: "train",
    },
    "MGTBench_train": {
        dataset_path: std.format("az://snlp-data/genaios_mgt_detection_attribution/%s/MGTBench/%s/", [language, task]),
        split: "train",
    },
    "m4_train": {
        dataset_path: std.format("az://snlp-data/genaios_mgt_detection_attribution/%s/m4/%s/", [language, task]),
        split: "train",
    },
    "genaios_generations_train": {
        dataset_path: std.format("az://snlp-data/genaios_mgt_detection_attribution/%s/genaios_generations/%s/", [language, task]),
        split: "train",
    },
};

local test_data = {
    "argugpt_test":{ 
        dataset_path: std.format("az://snlp-data/genaios_mgt_detection_attribution/%s/argugpt/%s/", [language, task]),
        split: "test",
    }, 
    "argugpt_validation": test_data.argugpt_test + {split: "validation"},
    "chatgpt_essays": {
        dataset_path: std.format("az://snlp-data/genaios_mgt_detection_attribution/%s/chatgpt-essays/%s/", [language, task]),
        split: "test",
    },
    "deepfake_test": train_data.deepfake_train + {split: "test"},
    "deepfake_test_ood_gpt": train_data.deepfake_train + {split: "test_ood_gpt"},
    "deepfake_test_ood_gpt_para": train_data.deepfake_train + {split: "test_ood_gpt_para"},
    "deepfake_validation": train_data.deepfake_train + {split: "validation"},
    "genaios_generations_test": train_data.genaios_generations_train + {split: "test"},
    "gpt4instructions_test": train_data.gpt4instructions_train + {split: "test"},
    "m4_test": train_data.m4_train + {split: "test"},
    "MGTBench_test": train_data.MGTBench_train + {split: "test"},
    "openwebtext_cross_model_test": {
        dataset_path: std.format("az://snlp-data/genaios_mgt_detection_attribution/%s/openwebtext-cross-model/%s/", [language, task]),
        split: "test",
    },
};

local model = {
    "feature_params": {
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
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
