import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    captions = data['Captions']
    sentiments = data['Sentiment']
    return captions, sentiments

def encode_sentiments(sentiments):
    label_encoder = LabelEncoder()
    encoded_sentiments = label_encoder.fit_transform(sentiments)
    return encoded_sentiments, label_encoder

def preprocess_text(captions):
    if isinstance(captions, pd.Series):
        captions = captions.tolist()
    elif isinstance(captions, (list, tuple)):
        pass
    else:
        raise ValueError("Input must be a list, tuple, or pandas Series.")

    inputs = tokenizer(captions, return_tensors='pt', padding=True, truncation=True, max_length=128)
    return inputs

def fine_tune_bert(bert_model, captions, sentiments, epochs=3):
    optimizer = AdamW(bert_model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    encoded_sentiments, label_encoder = encode_sentiments(sentiments)
    sentiments_tensor = torch.tensor(encoded_sentiments, dtype=torch.long)
    
    captions_train, captions_val, sentiments_train, sentiments_val = train_test_split(
        captions, sentiments_tensor, test_size=0.2, random_state=42)
    
    train_inputs = preprocess_text(captions_train)
    val_inputs = preprocess_text(captions_val)

    train_data = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], sentiments_train)
    val_data = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], sentiments_val)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    bert_model.train()
    for epoch in range(epochs):  # Increasing the number of epochs
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    bert_model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")



import torch.nn.functional as F

def predict_sentiment(captions):
    # Load the model
    model = BertForSequenceClassification.from_pretrained("saved_model")
    model.eval()

    # Preprocess input captions
    inputs = preprocess_text(captions)

    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        print("Logits:", logits)  # Debug: Check the raw output of the model

        predictions = torch.argmax(logits, dim=-1)

    return predictions.numpy()


def train_model():
    file_path = 'instasentiment.csv'  # Path to your dataset
    captions, sentiments = load_and_preprocess_data(file_path)
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Ensure num_labels=3
    fine_tune_bert(bert_model, captions, sentiments)

if __name__ == "__main__":
    train_model()
