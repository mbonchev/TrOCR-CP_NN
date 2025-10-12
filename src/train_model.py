import os
from typing import Tuple, List
import pandas as pd
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    default_data_collator
)
from PIL import Image
from datasets import Dataset

# Testing purpose parameters
FAST_TEST = True

import torch
# print("PyTorch is using", torch.get_num_threads(), "threads") # 6
torch.set_num_threads(4)

'''
# За жалост не мога да обуча модела на моята видеокарта (RX 580) и трябва да използвам CPU-то :(
import torch
print("PyTorch version:", torch.__version__)
print("ROCm available:", torch.version.hip is not None)
print("HIP (AMD) version:", torch.version.hip)
'''
'''
import kagglehub
# Download latest dataset version
path = kagglehub.dataset_download("preatcher/standard-ocr-dataset")
print("Path to dataset files: ", path)
'''
'''
import transformers
print(transformers.__version__) # v4.57.0
'''

# Collect image paths and corresponding labels
def load_image_labels(dataset_path: str, extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> Tuple[List[str], List[str]]:
    image_paths = []
    labels = []

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(extensions):
                    image_paths.append(os.path.join(folder_path, img_file))
                    labels.append(folder_name)  # Folder name is the label

    return image_paths, labels

def validate_and_subset_data(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels):
    # Check lengths of paths vs labels
    assert len(train_paths) == len(train_labels), f"Train mismatch: {len(train_paths)} images vs {len(train_labels)} labels"
    assert len(val_paths) == len(val_labels), f"Validation mismatch: {len(val_paths)} images vs {len(val_labels)} labels"
    assert len(test_paths) == len(test_labels), f"Test mismatch: {len(test_paths)} images vs {len(test_labels)} labels"

    # Use smaller subset if FAST_TEST is True
    train_subset_size = min(150, len(train_paths)) if FAST_TEST else len(train_paths)
    val_subset_size = min(30, len(val_paths)) if FAST_TEST else len(val_paths)
    test_subset_size = min(30, len(test_paths)) if FAST_TEST else len(test_paths)

    return train_subset_size, val_subset_size, test_subset_size

# Preprocessing function
def preprocess(batch):
    images = [Image.open(path).convert("RGB") for path in batch["image_path"]]
    # Convert images and labels to model inputs
    inputs = processor(images=images, text=batch["text"], return_tensors="pt", padding=True)
    # Add the processed tensors to the batch
    batch["pixel_values"] = inputs["pixel_values"] # The image tensors for the model
    batch["labels"] = inputs["labels"] # The tokenized text for training
    return batch

# Load image paths and labels
# training
train_data_path = "datasets/data/training_data"
train_paths, train_labels = load_image_labels(train_data_path)
# validation
val_data_path   = "datasets/data/validation_data"
val_paths, val_labels     = load_image_labels(val_data_path)
# testing
test_data_path  = "datasets/data/testing_data"
test_paths, test_labels   = load_image_labels(test_data_path)

train_subset_size, val_subset_size, test_subset_size = validate_and_subset_data(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)

print ("Train:", len(train_paths), "| Validation:", len(val_paths), "| Test:", len(test_paths))

# Load processor AND model
model_name = "microsoft/trocr-large-printed"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
# Model config
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.loss_type = "ForCausalLMLoss"

# Create HuggingFace datasets
# training
train_dict = ({"image_path": train_paths[:train_subset_size], "text": train_labels[:train_subset_size]})
train_dataset = Dataset.from_dict(train_dict).map(preprocess, batched=True, batch_size=4, remove_columns=["image_path", "text"])
# validation
val_dict = ({"image_path": val_paths[:val_subset_size], "text": val_labels[:val_subset_size]})
val_dataset = Dataset.from_dict(val_dict).map(preprocess, batched=True, batch_size=4, remove_columns=["image_path", "text"])
# testing
test_dict = ({"image_path": test_paths[:test_subset_size], "text": test_labels[:test_subset_size]})
test_dataset = Dataset.from_dict(test_dict).map(preprocess, batched=True, batch_size=4, remove_columns=["image_path", "text"])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    predict_with_generate=False, # For faster evaluation, due to the HW limitations
    logging_steps=10, # Log training metrics (to the console)
    save_steps=50, # Save a checkpoint
    eval_strategy="epoch", # "eval", NOT "evaluation"
    eval_steps=50,
    num_train_epochs=3, # Number of times to iterate through the dataset
    fp16=False,  # True = Half-precision -> faster with less memory on GPU, BUT since we are poor and use CPU, FP32 is the go-to
    save_total_limit=2  # Keep only the last 2 checkpoints to save disk space
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator, # Merges multiple dataset examples into one batch
    # tokenizer=processor.tokenizer
    processing_class=processor
)
# Start Training
trainer.train()

# Save Model
model.save_pretrained("./trocr-finetuned")
processor.save_pretrained("./trocr-finetuned")

# Evaluate on unseen test data
pred_ids = trainer.predict(test_dataset).predictions
pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
for i in range(5):
    print("Ground Truth:", test_labels[i], " | Prediction:", pred_texts[i])