import spacy
import json
import random
import os
from spacy.training.example import Example
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

TRAIN_DATA_PATH = '/teamspace/studios/this_studio/output/output_samples/sample_data_4.jsonl'
OUTPUT_MODEL_PATH = '/teamspace/studios/this_studio/Model/spacy/spacy_iteration_004'

if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

ner.add_label("ASPECT")

def load_training_data(path):
    training_data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            sentence = data['sentence']
            entities = []
            entity = data['entities']
            try:
                start_offset = int(entity['start_offset']) if entity['start_offset'] else None
                end_offset = int(entity['end_offset']) if entity['end_offset'] else None
                if start_offset is not None and end_offset is not None:
                    entities.append((start_offset, end_offset, "ASPECT"))
            except ValueError:
                print(f"Skipping entity with invalid offsets: {entity}")
                continue
            training_data.append((sentence, {"entities": entities}))
    return training_data

train_data = load_training_data(TRAIN_DATA_PATH)

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    def get_examples():
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            yield Example.from_dict(doc, annotations)

    nlp.initialize(get_examples)

    for epoch in range(10):
        random.shuffle(train_data)  
        losses = {}
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, losses=losses)

        print(f"Epoch {epoch + 1}: Losses {losses}")

os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
nlp.to_disk(OUTPUT_MODEL_PATH)
print(f"Model saved at {OUTPUT_MODEL_PATH}")