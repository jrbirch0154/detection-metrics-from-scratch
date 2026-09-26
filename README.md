# Detection Metrics from Scratch

IoU, non-maximum suppression, detection matching, and precision/recall/F1/AP for object detection, implemented from scratch in NumPy.

The notebook builds the standard object-detection evaluation pipeline one stage at a time, explaining each step before implementing it, then runs the full pipeline on an example scene.

## Contents

1. **Intersection over Union (IoU):** vectorized pairwise IoU between two sets of boxes, with input validation.
2. **Non-Maximum Suppression (NMS):** removes duplicate detections of the same object.
3. **Detection Matching:** greedy, confidence-ordered matching of predictions to ground truth (COCO-style), producing TP/FP/FN.
4. **Precision, Recall, and F1:** metrics at a single operating point.
5. **Average Precision (AP):** all-point interpolated area under the precision-recall curve (PASCAL VOC 2010+ method).
6. **Full Pipeline Example:** a 4-object scene with duplicates, false alarms, a badly localized box, and a missed object, evaluated with and without NMS.

## Example Results

| | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| Without NMS | 0.333 | 0.750 | 0.462 | 0.594 |
| With NMS | 0.500 | 0.750 | 0.600 | 0.650 |

NMS removes duplicate boxes, raising precision without losing recall.

## Running

Requires Python 3.10+ and NumPy. (Reference requirements.txt)
```
pip install -r requirements.txt
jupyter notebook IOU_Full_Stack.ipynb
```


## Conventions

- Boxes are `[x1, y1, x2, y2]` with `x1 <= x2` and `y1 <= y2`. Flipped boxes raise a `ValueError`.
- The default IoU threshold is 0.5 for both NMS and matching.
- Each ground-truth box can be matched at most once. Duplicate detections count as false positives.

## License

MIT
