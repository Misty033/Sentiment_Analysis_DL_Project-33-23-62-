import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import json
from tqdm import tqdm

# ---------------- Model ----------------
class AspectAttentionBERT(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', num_classes=3, num_heads=4, dropout_prob=0.3):
        super(AspectAttentionBERT, self).__init__()
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

# ---------------- Dataset ----------------
class AspectSentimentDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    sentence = record.get("sentence")
                    sentiment_dict = record.get("sentiment_dict", {})
                    aspect = record.get("entities", {}).get("aspect")   
                    if aspect and isinstance(sentiment_dict, dict):
                        sentiment_value = sentiment_dict.get(aspect, 0.0)  
                        sentiment_label = 0  
                        if sentiment_value < 0:
                            sentiment_label = 0  # Negative
                        elif sentiment_value == 0:
                            sentiment_label = 1  # Neutral
                        elif sentiment_value > 0:
                            sentiment_label = 2  # Positive
                        self.data.append({
                            "sentence": sentence,
                            "aspect": aspect,
                            "label": sentiment_label
                        })
                except Exception as e:
                    print(f"Skipping line due to error: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        sentence = row['sentence']
        aspect = row['aspect']
        label = row['label']
        inputs = self.tokenizer.encode_plus(
            text=aspect,
            text_pair=sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ---------------- Training Function ----------------
def train(model, dataloader, epochs=10, lr=2e-5):
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader)}")

# ---------------- Save Model ----------------
def save_model(model, path='/teamspace/studios/this_studio/Model/aspect_sentiment_model04.pt'):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")

if __name__ == '__main__':
    JSONL_PATH = '/teamspace/studios/this_studio/output/output_samples/sampled_data_4.jsonl'
    PRETRAINED_MODEL = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
    dataset = AspectSentimentDataset(JSONL_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AspectAttentionBERT(pretrained_model=PRETRAINED_MODEL).to(device)
    train(model, dataloader)
    save_model(model)