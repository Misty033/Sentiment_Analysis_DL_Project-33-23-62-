import torch
import torch.nn as nn
from transformers import BertModel

class AspectMultiheadAttentionBERT(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', num_classes=3, num_heads=4, dropout_prob=0.3):
        super(AspectMultiheadAttentionBERT, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size

        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_token = hidden_states[:, 0:1, :]

        attn_output, _ = self.attention_layer(
            query=cls_token,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=~attention_mask.bool()
        )

        context_vector = self.layer_norm(attn_output.squeeze(1))
        logits = self.classifier(context_vector)
        return logits

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
        return predicted_class.item(), probs.detach().cpu().numpy().flatten()