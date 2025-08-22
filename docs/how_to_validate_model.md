# How to Validate Your Model

## 1. Split Your Data
- Divide your labeled data into training, validation, and (optionally) test sets.
- Typical split: 70% train, 15% val, 15% test.

## 2. Use Validation During Training
- After each training epoch, run your model on the validation set.
- Track metrics: accuracy, precision, recall, F1 score.
- Example (see template):
  - `validate(model, val_loader, device)` prints validation accuracy.

## 3. Test on Unseen Data
- After training, evaluate your model on the test set (never used for training).
- Compare metrics to validation results.

## 4. Check Confusion Matrix
- For classification tasks, plot a confusion matrix to see which classes are confused.

## 5. Review Predictions
- Manually inspect some predictions to ensure the model is making sensible decisions.

## 6. Iterate
- If accuracy is low, try:
  - More data
  - Data augmentation
  - Different model architectures
  - Hyperparameter tuning

## 7. Save Metrics and Model
- Save your trained model and metrics for future reference.

## 8. Document Everything
- Keep notes on your data, training process, and results.

---
**Tip:** Always use held-out data for validation/testing. Never evaluate on the same data you train on.
