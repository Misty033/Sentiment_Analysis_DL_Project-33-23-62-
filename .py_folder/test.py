import json
import torch
from spacy_inference import SpacyAspectExtractor
from model_ import AspectMultiheadAttentionBERT
from transformers import BertTokenizer

TEST_FILE = '/teamspace/studios/this_studio/test2.jsonl'
SPACY_MODEL_PATH = '/teamspace/studios/this_studio/Model/spacy/spacy_iteration_004'
MODEL_PATH = '/teamspace/studios/this_studio/Model/aspect_sentiment_model04.pt'
PRETRAINED_MODEL = 'bert-base-uncased'

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
model = AspectMultiheadAttentionBERT(PRETRAINED_MODEL)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

spacy_extractor = SpacyAspectExtractor(SPACY_MODEL_PATH)
extracted_aspects_all = []
extracted_labels_all = []
ground_truth_aspects_all = []
ground_truth_labels_all = []

def get_sentiment_label_from_dict(sent_dict):
    if not sent_dict:
        return 0
    label_value = list(sent_dict.values())[0]
    return int(label_value)

def encode_input_with_aspect(sentence, aspect, tokenizer, max_length=128):
    inputs = tokenizer.encode_plus(
        text=aspect,
        text_pair=sentence,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return inputs['input_ids'], inputs['attention_mask']

with open(TEST_FILE, 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        sentence = item['sentence']
        gt_entity = item['entities']
        sent_dict = item['sentiment_dict']
        gt_aspect = (gt_entity['aspect'], gt_entity['start_offset'], gt_entity['end_offset'])
        gt_label = get_sentiment_label_from_dict(sent_dict)
        ground_truth_aspects_all.append([gt_aspect])
        ground_truth_labels_all.append([gt_label])
        doc = spacy_extractor.nlp(sentence)
        extracted_aspect_triplets = []
        pred_labels = []

        for ent in doc.ents:
            asp_text = ent.text
            start = ent.start_char
            end = ent.end_char
            if start is None or end is None:
                continue
            extracted_aspect_triplets.append((asp_text, start, end))
            input_ids, attention_mask = encode_input_with_aspect(sentence, asp_text, tokenizer)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_labels.append(pred_class)

        extracted_aspects_all.append(extracted_aspect_triplets)
        extracted_labels_all.append(pred_labels)