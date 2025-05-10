# ------------model_.py-------------
from model_ import AspectMultiheadAttentionBERT
import torch
import json
from tqdm import tqdm
from transformers import BertTokenizer

# -----------BERT INFERENCE----------- 
def bert_inference(
    input_jsonl_path,
    output_jsonl_path,
    bert_model_path,
    pretrained_model='bert-base-uncased'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = AspectMultiheadAttentionBERT(pretrained_model)

    try:
        model.load_state_dict(torch.load(bert_model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    try:
        with open(input_jsonl_path, 'r') as infile, open(output_jsonl_path, 'w') as outfile:
            for line in tqdm(infile, desc="Running BERT Inference", unit="line"):
                try:
                    item = json.loads(line)
                    sentence = item.get("sentence", "")
                    aspects = item.get("aspects", [])

                    if not sentence or not aspects:
                        continue

                    aspect_sentiments = {}

                    for aspect in aspects:
                        if not aspect:
                            continue

                        inputs = tokenizer.encode_plus(
                            text=aspect,
                            text_pair=sentence,
                            add_special_tokens=True,
                            max_length=128,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )

                        input_ids = inputs['input_ids'].to(device)
                        attention_mask = inputs['attention_mask'].to(device)

                        with torch.no_grad():
                            pred_class, probs = model.predict(input_ids, attention_mask)

                        confidence_score = max(probs)

                        aspect_sentiments[aspect] = {
                            "prob_dict": {i: float(p) for i, p in enumerate(probs)},
                            "sentiment": pred_class,
                            "confidence_score": float(confidence_score)
                        }

                    item["aspect_sentiments"] = aspect_sentiments
                    outfile.write(json.dumps(item) + '\n')

                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue

    except Exception as e:
        print(f"Error opening files: {e}")


# -----------PATH----------- 
if __name__ == "__main__":
    input_jsonl_path = "/teamspace/studios/this_studio/Inferences/screen_protector_data.jsonl"      #input data
    output_jsonl_path = "/teamspace/studios/this_studio/Inferences/screen_protector_data_TS.jsonl"  #output data
    bert_model_path = "/teamspace/studios/this_studio/Model/aspect_sentiment_model04.pt"            #weights of the last iteration

    bert_inference(
        input_jsonl_path=input_jsonl_path,
        output_jsonl_path=output_jsonl_path,
        bert_model_path=bert_model_path,
        pretrained_model='bert-base-uncased'
    )
