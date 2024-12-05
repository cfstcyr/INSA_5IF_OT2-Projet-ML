# Test runs

## Test run 1

- **model**: `model_2024-12-04_20-29-44`
    - n_epochs: 2
    - val_loss:
        - 0.180
        - 0.092
- **results**: `results_2024-12-04_20-32-16`
    - Accuracy: 88.86 %
        - noface: 97.31 %
        - face: 16.44 %

## Test run 2

- **model**: `model_2024-12-04_20-34-56`
    - n_epochs: 10
    - val_loss:
        - 0.241
        - 0.114 (-0.127)
        - 0.074 (-0.040)
        - 0.067 (-0.007)
        - 0.041 (-0.026)
        - 0.038 (-0.003)
        - 0.030 (-0.008)
        - 0.031 (+0.001)
        - 0.030 (-0.001)
        - 0.019 (-0.011)
- **results**: `results_2024-12-04_20-39-48` (compared to `results_2024-12-04_20-32-16`)
    - Accuracy: 91.81 % (+2.95 %)
        - noface: 99.09 % (+1.78 %)
        - face: 29.36 % (+12.92 %)

> Great improvement in face detection accuracy

## Test run 3

- **model**: `model_2024-12-04_20-41-56`
    - n_epochs: 50
    - val_loss:
        - **0**: 0.171
        - **10**: 0.025 (-0.146)
        - **20**: 0.012 (-0.013)
        - **30**: 0.014 (+0.002)
        - **40**: 0.010 (-0.004)
        - **50**: 0.011 (+0.001)
- **results**: `results_2024-12-04_21-08-59` (compared to `results_2024-12-04_20-39-48`)
    - Accuracy: 94.22 % (+2.41 %)
        - noface: 99.62 % (+0.53 %)
        - face: 47.93 % (+18.57 %)

> Great improvement in face detection accuracy

> By plotting the loss curve, we can see that the model works well between 10 and 20 epochs, and then starts to overfit. We will use 20 epochs for the final model.

## Test run 4

- **model**: `model_2024-12-04_21-15-36`
    - n_epochs: 20
- **results**: `results_2024-12-04_22-01-37` (compared to `results_2024-12-04_21-08-59`)
    - Accuracy: 94.35 % (+0.13 %)
        - noface: 99.53 % (-0.09 %)
        - face: 49.94 % (+2.01 %)

## Test run 5

- **model**: `model_2024-12-04_22-03-26`
    - n_epochs: 20
    - transforms:
        - `transforms.RandomHorizontalFlip(p=0.5)`
- **results**: `results_2024-12-04_22-30-28` (compared to `results_2024-12-04_22-01-37`)
    - Accuracy: 93.46 % (-0.89 %)
        - noface: 99.41 % (-0.12 %)
        - face: 42.41 % (-7.53 %)

> Image augmentation with image flip is not useful for this dataset

## Test run 6

- **model**: `model_2024-12-05_11-54-56`
    - n_epochs: 20
    - transforms:
        - `transforms.RandomCrop(32, padding=4)`
- **results**: `results_2024-12-05_12-07-10` (compared to `results_2024-12-04_22-30-28`)
    - Accuracy: 96.46 % (+3.00 %)
        - noface: 98.01 % (-1.40 %)
        - face: 83.19 % (+40.78 %)

> Great improvement in face detection accuracy with image cropping

## Test run 7

- **model**: `model_2024-12-05_11-33-59`
    - n_epochs: 20
    - transforms:
        - `transforms.RandomHorizontalFlip(p=0.5)`
        - `transforms.RandomCrop(32, padding=4)`
- **results**: `results_2024-12-05_11-51-04` (compared to `results_2024-12-05_12-07-10`)
    - Accuracy: 97.40 %  (+0.94 %)
        - noface: 99.25 % (+1.24 %)
        - face: 81.56 % (-1.63 %)

> Slight improvement loss of face detection accuracy with flip and crop, but overall improvement in accuracy