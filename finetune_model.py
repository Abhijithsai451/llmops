import os

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()
def download_model():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set. Please set it in your .env file or environment.")

    model_id = os.getenv("MODEL_ID")
    local_dir = os.getenv("MODEL_DIR")

    print(f"Attempting to download {model_id} to {local_dir}...")

    try:
        snapshot_download(repo_id=model_id, local_dir=local_dir, token=hf_token)
        print(f" Model {model_id} successfully downloaded to {local_dir}")
    except Exception as e:
        print(f" Error downloading model: {e}")
        print("Please ensure you have access to the model on Hugging Face and your HF_TOKEN is valid.")


def create_model():
    load_dotenv()
    model_id = os.getenv("MODEL_ID")
    local_dir = os.getenv("MODEL_DIR")
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(local_dir)


    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    dataset = dataset['train'].select(range(2000))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    tokenized_dataset.set_format(type="torch")

    # Add labels (important for training)
    def add_labels(example):
        example["labels"] = example["input_ids"].clone()
        return example

    tokenized_dataset = tokenized_dataset.map(add_labels)

    # Training config
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    # Train
    trainer.train()

    # Save model
    torch.save(model.state_dict(), "model.pt")

    print("Model saved as model.pt")





if __name__ == "__main__":
    download_model()
    create_model()