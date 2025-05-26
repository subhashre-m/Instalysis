import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv('dataset_captions.csv')

# Mapping mental states to numerical labels
label_mapping = {
    "Normal": 0,
    "Happy": 1,
    "Stressed": 2,
    "Moderately Depressed": 3,
    "Severely Depressed": 4
}

# Apply the label mapping
df['mental_state'] = df['mental_state'].map(label_mapping)

# Split into train and test datasets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert the dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[['caption', 'mental_state']])
val_dataset = Dataset.from_pandas(val_df[['caption', 'mental_state']])

# Load XLM-RoBERTa tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', local_files_only=False)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['caption'], padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Load XLM-RoBERTa model for sequence classification
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=5)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",              # Output directory for model predictions and checkpoints
    per_device_train_batch_size=8,       # Batch size for training
    per_device_eval_batch_size=8,        # Batch size for evaluation
    num_train_epochs=3,                  # Number of training epochs
    logging_dir="./logs",                # Directory for storing logs
    evaluation_strategy="epoch",         # Evaluate after each epoch
    save_strategy="epoch",               # Save checkpoint after each epoch
    logging_steps=10,
    save_steps=1000,
    load_best_model_at_end=True,         # Load the best model after training
    fp16=True,                           # Use mixed precision for faster training
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./finetuned_model')
tokenizer.save_pretrained('./finetuned_model')

print("Fine-tuning complete and model saved.")
