import torch
import torch.nn as nn
from transformers import AutoModel

class FusionConfig:
    model_type = "fusion_classifier"
    
    def __init__(self, num_labels=5, xlmr_checkpoint="xlm-roberta-large", mbert_checkpoint="bert-base-multilingual-cased", xlmr_hidden_size=1024, mbert_hidden_size=768, label2id=None, id2label=None):
        self.num_labels = num_labels
        self.xlmr_checkpoint = xlmr_checkpoint
        self.mbert_checkpoint = mbert_checkpoint
        self.xlmr_hidden_size = xlmr_hidden_size
        self.mbert_hidden_size = mbert_hidden_size
        self.combined_hidden_size = xlmr_hidden_size + mbert_hidden_size
        self.label2id = label2id
        self.id2label = id2label

    def to_dict(self):
        return {
            "_name_or_path": "xlm_bert_mental_state_classifier",
            "model_type": self.model_type,
            "num_labels": self.num_labels,
            "xlmr_checkpoint": self.xlmr_checkpoint,
            "mbert_checkpoint": self.mbert_checkpoint,
            "xlmr_hidden_size": self.xlmr_hidden_size,
            "mbert_hidden_size": self.mbert_hidden_size,
            "combined_hidden_size": self.combined_hidden_size,
            "label2id": self.label2id,
            "id2label": self.id2label
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            num_labels = config_dict["num_labels"],
            xlmr_checkpoint = config_dict["xlmr_checkpoint"],
            mbert_checkpoint = config_dict["mbert_checkpoint"],
            xlmr_hidden_size = config_dict["xlmr_hidden_size"],
            mbert_hidden_size = config_dict["mbert_hidden_size"],
            label2id = config_dict["label2id"],
            id2label = config_dict["id2label"]
        )

class FusionClassifier(nn.Module):
    def __init__(self, config, dropout_rate=0.1):
        super(FusionClassifier, self).__init__()
        # Load pretrained encoders without classification heads.
        self.xlmr = AutoModel.from_pretrained(config.xlmr_checkpoint)
        self.mbert = AutoModel.from_pretrained(config.mbert_checkpoint)
        
        # For XLM-R, we extract the [CLS] token representation.
        # For mBERT, we use mean pooling.
        self.fc1 = nn.Linear(config.xlmr_hidden_size, config.xlmr_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        combined_size = config.xlmr_hidden_size + config.mbert_hidden_size
        self.fc2 = nn.Linear(combined_size, config.num_labels)
    
    def forward(self, xlmr_input_ids, xlmr_attention_mask, mbert_input_ids, mbert_attention_mask, labels=None):
        # XLM-R branch: get [CLS] token and transform it.
        xlmr_outputs = self.xlmr(input_ids=xlmr_input_ids, attention_mask=xlmr_attention_mask)
        xlmr_cls = xlmr_outputs.last_hidden_state[:, 0, :]
        transformed_xlmr = self.relu(self.fc1(xlmr_cls))
        
        # mBERT branch: apply mean pooling over token embeddings.
        mbert_outputs = self.mbert(input_ids=mbert_input_ids, attention_mask=mbert_attention_mask)
        mbert_mean = torch.mean(mbert_outputs.last_hidden_state, dim=1)
        
        # Fusion: concatenate and pass through dropout and final FC layer.
        fused = torch.cat([transformed_xlmr, mbert_mean], dim=1)
        fused = self.dropout(fused)
        logits = self.fc2(fused)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
