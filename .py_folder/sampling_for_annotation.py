import json
import pandas as pd
import uuid

def map_sentiment_label(label):
    if label == -1:
        return 0
    elif label == 0:
        return 1
    elif label == 1:
        return 2
    return label

def sample_and_split_low_confidence(input_jsonl_path, sampled_output_path, remaining_output_path, sample_size=100):
    df = pd.read_json(input_jsonl_path, lines=True)

    if 'id' not in df.columns:
        df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]

    df['confidence_score'] = df['aspect_sentiments'].apply(
        lambda x: sum([aspect['confidence_score'] for aspect in x.values()]) / len(x) if isinstance(x, dict) else None
    )

    sampled_df = df.nsmallest(sample_size, 'confidence_score').reset_index(drop=True)
    sampled_ids = set(sampled_df['id'])
    remaining_df = df[~df['id'].isin(sampled_ids)].reset_index(drop=True)
    sampled_df.to_json(sampled_output_path, orient='records', lines=True)
    remaining_df.to_json(remaining_output_path, orient='records', lines=True)
    print(f"Sampled entries to: {sampled_output_path}")
    print(f"Remaining entries saved to: {remaining_output_path}")

input_jsonl = '/teamspace/studios/this_studio/Inferences/bert_inference_output4.jsonl'
sampled_output = '/teamspace/studios/this_studio/sampled_data_fol/low_confidence_samples_04.jsonl'
remaining_output = '/teamspace/studios/this_studio/remaining_points/after_sampling_4.jsonl'
sample_size = 100
sample_and_split_low_confidence(input_jsonl, sampled_output, remaining_output, sample_size)