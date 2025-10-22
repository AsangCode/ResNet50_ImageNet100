# ResNet50 Training Log

## Learning Rate Finder Results
- Minimum loss achieved at learning rate: 0.966051
- Suggested learning rate (min_loss_lr/10): 0.096605

## Training Progress

### Epoch 1/15
- Training:
  - Loss: 4.4409
  - Accuracy: 4.04%
- Validation:
  - Loss: 4.0663
  - Accuracy: 8.62%

### Epoch 2/15
- Training:
  - Loss: 3.7723
  - Accuracy: 12.43%
- Validation:
  - Loss: 3.3902
  - Accuracy: 17.48%

### Epoch 3/15
- Training:
  - Loss: 3.3563
  - Accuracy: 19.52%
- Validation:
  - Loss: 3.1204
  - Accuracy: 23.62%

### Epoch 4/15
- Training:
  - Loss: 3.0547
  - Accuracy: 25.49%
- Validation:
  - Loss: 2.9967
  - Accuracy: 27.90%

### Epoch 5/15
- Training:
  - Loss: 2.8255
  - Accuracy: 30.01%
- Validation:
  - Loss: 2.7736
  - Accuracy: 32.28%

### Epoch 6/15
- Training:
  - Loss: 2.6243
  - Accuracy: 34.57%
- Validation:
  - Loss: 2.4351
  - Accuracy: 37.14%

### Epoch 7/15
- Training:
  - Loss: 2.4401
  - Accuracy: 38.56%
- Validation:
  - Loss: 2.0853
  - Accuracy: 44.62%

### Epoch 8/15
- Training:
  - Loss: 2.2889
  - Accuracy: 41.88%
- Validation:
  - Loss: 2.2410
  - Accuracy: 43.58%

### Epoch 9/15
- Training:
  - Loss: 2.1590
  - Accuracy: 44.89%
- Validation:
  - Loss: 1.9679
  - Accuracy: 48.90%

### Epoch 10/15
- Training:
  - Loss: 2.0388
  - Accuracy: 47.82%
- Validation:
  - Loss: 2.0474
  - Accuracy: 49.02%

### Epoch 11/15
- Training:
  - Loss: 1.9464
  - Accuracy: 49.81%
- Validation:
  - Loss: 1.8256
  - Accuracy: 52.64%

### Epoch 12/15
- Training:
  - Loss: 1.8620
  - Accuracy: 51.82%
- Validation:
  - Loss: 1.6620
  - Accuracy: 55.20%

### Epoch 13/15
- Training:
  - Loss: 1.7814
  - Accuracy: 53.77%
- Validation:
  - Loss: 1.5798
  - Accuracy: 56.60%

### Epoch 14/15
- Training:
  - Loss: 1.7227
  - Accuracy: 55.21%
- Validation:
  - Loss: 1.6558
  - Accuracy: 56.44%

### Epoch 15/15
- Training:
  - Loss: 1.6666
  - Accuracy: 56.64%
- Validation:
  - Loss: 1.5490
  - Accuracy: 58.26%

## Training Statistics
- Training iterations per epoch: 990
- Validation iterations per epoch: 40
- Average training speed: ~2.3 iterations/second
- Average validation speed: ~2.5 iterations/second

## System Notes
- Using CUDA with TF32 precision
- Warning: Not enough SMs for max_autotune_gemm mode
- Learning rate scheduler warning: Called before optimizer step (expected behavior)
