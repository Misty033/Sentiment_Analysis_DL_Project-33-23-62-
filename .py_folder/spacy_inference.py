import spacy
import pandas as pd
import json

class SpacyAspectExtractor:
    def __init__(self, spacy_model_path):
        self.nlp = spacy.load(spacy_model_path)

    def extract(self, sentence):
        doc = self.nlp(sentence)
        entities = [
            {
                "aspect": ent.text,
                "start_offset": ent.start_char,
                "end_offset": ent.end_char
            }
            for ent in doc.ents
        ]
        aspects = ", ".join(ent["aspect"] for ent in entities)
        return {
            "sentence": sentence,
            "entities": entities,
            "aspect": aspects
        }

    def process_dataframe(self, df, text_column="text"):
        results = []
        for sentence in df[text_column].dropna().astype(str):
            result = self.extract(sentence.strip())
            results.append(result)
        return results

    def save_results(self, results, output_file):
        with open(output_file, "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    spacy_model_path = "/teamspace/studios/this_studio/Model/spacy/spacy_iteration_004"
    input_parquet = "/teamspace/studios/this_studio/sampled_data.parquet"
    output_jsonl = "/teamspace/studios/this_studio/Inferences/spacy_aspects_4.jsonl"
    extractor = SpacyAspectExtractor(spacy_model_path)
    df = pd.read_parquet(input_parquet)
    results = extractor.process_dataframe(df, text_column="text")
    extractor.save_results(results, output_jsonl)

    print(f"Saved extracted aspects to {output_jsonl}")