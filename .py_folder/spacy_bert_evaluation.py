# --- Evaluation ---
def calculate_iou(span1, span2):
    if None in span1 or None in span2:
        return 0.0
    start1, end1 = int(span1[0]), int(span1[1])
    start2, end2 = int(span2[0]), int(span2[1])
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union != 0 else 0.0

def evaluate_aspect_extraction(pred_aspects_all, gt_aspects_all, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = 0

    for pred_list, gt_list in zip(pred_aspects_all, gt_aspects_all):
        matched = set()
        for pred in pred_list:
            pred_text, pred_start, pred_end = pred
            if pred_start is None or pred_end is None:
                continue
            matched_flag = False
            for idx, gt in enumerate(gt_list):
                gt_text, gt_start, gt_end = gt
                if gt_start is None or gt_end is None:
                    continue
                iou = calculate_iou((gt_start, gt_end), (pred_start, pred_end))
                if iou >= iou_threshold:
                    tp += 1
                    matched.add(idx)
                    matched_flag = True
                    break
            if not matched_flag:
                fp += 1
        fn += len(gt_list) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

# --- Aspect Extraction Evaluation ---
print("\n--- Aspect Extraction Evaluation ---\n")
precision, recall, f1 = evaluate_aspect_extraction(extracted_aspects_all, ground_truth_aspects_all) #from spacy
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# --- Sentiment Classification Evaluation ---
from sklearn.metrics import classification_report

matched_preds = []
matched_trues = []

for pred_triplets, pred_labels, gt_triplets, gt_labels in zip(
    extracted_aspects_all, extracted_labels_all, ground_truth_aspects_all, ground_truth_labels_all): #from bert

    for pred_triplet, pred_label in zip(pred_triplets, pred_labels):
        pred_text, pred_start, pred_end = pred_triplet
        if pred_start is None or pred_end is None:
            continue
        for idx, gt_triplet in enumerate(gt_triplets):
            gt_text, gt_start, gt_end = gt_triplet
            if gt_start is None or gt_end is None:
                continue
            iou = calculate_iou((pred_start, pred_end), (gt_start, gt_end))
            if iou >= 0.5:
                matched_preds.append(pred_label)
                matched_trues.append(gt_labels[idx])
                break

if matched_preds:
    print("\n--- Sentiment Classification Evaluation ---\n")
    print("Classification Report:")
    print(classification_report(matched_trues, matched_preds, digits=4))
else:
    print("\nNo matching aspects found for sentiment classification.")
