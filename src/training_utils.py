from enum import Enum
import os
import torch
from datasets import DatasetDict, load_dataset, load_from_disk, Dataset
from datasets.builder import DatasetGenerationError
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import warnings
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from datasets.builder import DatasetGenerationError
import json

DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

def create_and_prepare_model(args, data_args, training_args):
    device_map = None
    bnb_config = None

    load_in_8bit = args.use_8bit_quantization
    load_in_4bit = args.use_4bit_quantization

    if args.use_unsloth:
        from unsloth import FastLanguageModel

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_stype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_stype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization or args.use_8bit_quantization:
        device_map = (
            int(os.environ.get("LOCAL_RANK", -1))
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else "auto"
        )  # {"": 0}

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=load_in_8bit,
            quantization_config=bnb_config,
            device_map=None,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=quant_storage_stype or torch.float32,
        )

    if (
        (args.use_4bit_quantization or args.use_8bit_quantization)
        and args.use_peft_lora
        and not args.use_unsloth
    ):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": args.use_reentrant},
        )

    peft_config = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )
        model = get_peft_model(model, peft_config)
    
    elif args.use_peft_lora and args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    special_tokens = None
    chat_template = None

    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        # make embedding resizing configurable?
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, padding_side="right"
        )
        tokenizer.pad_token = tokenizer.eos_token


    return model, peft_config, tokenizer 


def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    def formatting_func(example):
        EOS_TOKEN = tokenizer.eos_token
        return example + " " + EOS_TOKEN

    def preprocess(samples):
        return {"text":tokenizer.apply_chat_template(samples["messages"], tokenize=False)}

    if data_args.dataset_name == None and data_args.csv_path == None:
        raise ValueError("HF Dataset name or CSV path is required!")

    if data_args.dataset_name != None and data_args.csv_path != None:
        warnings.warn("Only one HF Dataset or CSV path can be used as data source, Defaulting to the HF Dataset.")
        data_args.csv_path = None

    if data_args.dataset_name != None:
        raw_datasets = DatasetDict()
        try:
                # Try first if dataset on a Hub repo
            dataset = load_dataset(data_args.dataset_name,split="train").train_test_split(test_size=0.1)
            
            train_data = dataset["train"]
            train_data = train_data.remove_columns(["__index_level_0__"])
            train_data_pd = train_data.to_pandas()
            train_data_pd.dropna(inplace=True)
            train_data = Dataset.from_pandas(train_data_pd)
            train_data = train_data.remove_columns(["__index_level_0__"])
            valid_data = dataset["test"]
            valid_data = valid_data.remove_columns(["__index_level_0__"])
            valid_data_pd = valid_data.to_pandas()
            valid_data_pd.dropna(inplace=True)
            valid_data = Dataset.from_pandas(valid_data_pd)
            valid_data = valid_data.remove_columns(["__index_level_0__"])

                
        except DatasetGenerationError:
                # If not, check local dataset
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

        if apply_chat_template:
            print("I AM HERE AHOLE")
            train_data = train_data.map(
                preprocess,
                batched=True,
                remove_columns=["messages"],
            )

            valid_data = valid_data.map(
                preprocess,
                batched=True,
                remove_columns=["messages"]
            )

    
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )
        print(f"A sample of train dataset: {train_data[0]}")

        return train_data, valid_data

    if data_args.csv_path != None:
        df = pd.read_csv(data_args.csv_path,engine="python",on_bad_lines='skip')
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        
        if data_args.append_concat_token == True:
            train_df[data_args.dataset_text_field] = train_df[data_args.dataset_text_field].apply(formatting_func)
            test_df[data_args.dataset_text_field] = test_df[data_args.dataset_text_field].apply(formatting_func)
        
        train_data = Dataset.from_pandas(train_df)
        if apply_chat_template:
            train_data = train_data.map(
                preprocess,
                remove_columns=["messages"],
            )
        valid_data = Dataset.from_pandas(test_df)
        if apply_chat_template:
            valid_data = valid_data.map(
                preprocess,
                remove_columns=["messages"],
            )
        print(train_data.head())
        return train_data,valid_data


