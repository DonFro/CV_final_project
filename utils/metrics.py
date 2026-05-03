import numpy as np

def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def compute_metrics(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0, 1.0, 1.0, 1.0
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0, 0.0

    tp = 0
    matched_gt = set()
    sorted_preds = sorted(zip(pred_scores, pred_boxes), reverse=True)
    precisions, recalls = [], []

    for i, (score, pred_box) in enumerate(sorted_preds):
        best_iou = 0
        best_gt_idx = -1
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)

        fp = (i + 1) - tp
        precisions.append(tp / (tp + fp))
        recalls.append(tp / len(gt_boxes))

    precision = precisions[-1] if precisions else 0
    recall = recalls[-1] if recalls else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    mAP = float(np.trapz(precisions, recalls)) if len(precisions) > 1 else precision

    return round(precision, 4), round(recall, 4), round(f1, 4), round(mAP, 4)