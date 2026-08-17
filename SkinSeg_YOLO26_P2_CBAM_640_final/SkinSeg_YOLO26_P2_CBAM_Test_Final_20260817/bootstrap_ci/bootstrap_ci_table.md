| Metric | Point | 95% bootstrap CI |
|---|---:|---:|
| top1_accuracy | 0.9037 | [0.8858, 0.9206] |
| box_iou_mean | 0.9093 | [0.9020, 0.9165] |
| box_iou50_rate | 0.9841 | [0.9762, 0.9911] |
| mask_iou_mean | 0.9038 | [0.8958, 0.9111] |
| mask_dice_mean | 0.9431 | [0.9366, 0.9492] |
| mask_iou50_rate | 0.9811 | [0.9732, 0.9891] |
| strict_class_and_mask_iou50_rate | 0.8878 | [0.8679, 0.9067] |
| macro_top1_accuracy | 0.8145 | [0.7682, 0.8595] |
| macro_mask_iou_mean | 0.8507 | [0.8266, 0.8708] |
| macro_mask_dice_mean | 0.9082 | [0.8865, 0.9251] |
| macro_strict_class_and_mask_iou50_rate | 0.7806 | [0.7247, 0.8334] |

## Notes for interpretation

These values are image-level nonparametric bootstrap estimates from 2000 resamples of the 1007-image test split. They are **not** bootstrap confidence intervals for Ultralytics COCO mAP. Official mAP values should still be reported from `test_metrics.json`.

`Point` is the metric computed on the full test split. `95% bootstrap CI` is the percentile interval obtained by resampling test images with replacement.

## Metric definitions

| Metric | Meaning | How to claim safely |
|---|---|---|
| `top1_accuracy` | Micro image-level class agreement between the highest-confidence prediction and the ground-truth class. Every image has equal weight, so majority classes strongly influence this number. | Use as an overall class-agreement proxy, but do not report it alone because NV dominates the test split. |
| `box_iou_mean` | Mean IoU between the predicted bounding box and the ground-truth polygon bounding box. | Use as a localization-quality proxy, not as COCO box mAP. |
| `box_iou50_rate` | Fraction of test images with predicted box IoU at least 0.50. | Can support the claim that most lesions are localized at a coarse IoU threshold. |
| `mask_iou_mean` | Mean pixel IoU between the predicted mask and the rasterized ground-truth polygon mask. | Use as an image-level mask-overlap proxy, not as mask mAP50-95. |
| `mask_dice_mean` | Mean Dice coefficient between predicted and ground-truth masks. Dice is usually higher than IoU for the same overlap. | Useful for medical-image readers, but explain it is derived from saved predictions. |
| `mask_iou50_rate` | Fraction of images with mask IoU at least 0.50. | Can support coarse mask-success discussion. |
| `strict_class_and_mask_iou50_rate` | Fraction of images where both class is correct and mask IoU is at least 0.50. | This is the safest single image-level success proxy because it requires both recognition and segmentation. |
| `macro_top1_accuracy` | Average top-1 accuracy computed equally across classes. Each class has equal weight regardless of sample count. | Prefer this over micro `top1_accuracy` when discussing class imbalance. |
| `macro_mask_iou_mean` | Average of per-class mean mask IoU. | Use to discuss segmentation quality without letting NV dominate. |
| `macro_mask_dice_mean` | Average of per-class mean Dice. | Use as a class-balanced Dice-style summary. |
| `macro_strict_class_and_mask_iou50_rate` | Average per-class strict success rate requiring correct class and mask IoU ≥ 0.50. | Best compact class-balanced proxy for joint class-aware segmentation success. |

## Safe claim wording

Recommended wording:

> To complement the mAP point estimates, image-level bootstrap analysis over 2000 resamples showed a mask Dice proxy of 0.9431 [0.9366, 0.9492] and a mask IoU proxy of 0.9038 [0.8958, 0.9111]. Because the test set is dominated by NV, class-balanced macro summaries were also reported: macro top-1 accuracy 0.8145 [0.7682, 0.8595] and macro strict class-and-mask-IoU50 success 0.7806 [0.7247, 0.8334].

Avoid:

> The mask mAP50-95 confidence interval is [0.8958, 0.9111].

This is incorrect because the bootstrap script does not recompute Ultralytics mAP for each bootstrap sample.

## Important caveat

The micro metrics are high partly because the test split contains many NV images. Use `per_class_metrics.csv` and the macro rows above when discussing class imbalance and difficult diagnoses such as MEL.
