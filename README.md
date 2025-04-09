# Efficient Adaptive Nonlinear Deep Coding with Visual State Space Model

This repository is the official implementation of [Adaptive Nonlinear Joint Source-Channel Coding with Visual State Space Model for Wireless Image Transmission].

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Training

To train MNTSCC on DIV2K dataset, run this command:

```train
python main.py --phase train --dataset kodak
```

## Evaluation

To evaluate my model on kodak, run:

```eval
python main.py --phase test --dataset kodak
```
