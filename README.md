# MemberInference-by-LossThreshold

This is a minimal implementation (in Pytorch 1.11) of [loss thresholding attack](https://arxiv.org/abs/1709.01604) to infer membership status (i.e., part of training data or not) of data samples.
Briefly, we have a threshold **T** (e.g., avg. training loss), and we predict a sample 'member' if sample loss < **T**.


Attack code is in ```src/attacks.py``` and some results on common image datasets (averaged over 5 runs) are as follows. Note that, baseline balanced acc. is 50% for MIA.

| Dataset | Model | Test Acc (%) | Generalization Gap (%) |  MIA Balanced Acc. (%)
| ------------- | ------------- | ------------- | ------------- | ------------- |
| MNIST  | LeNet5 |99.2 ± 0.05 | 0.76 ± 0.04 | 50.34% ± 0.1 |
| Fashion-MNIST | LeNet5 | 90.77 ± 0.11  | 5.02 ± 0.63 | 51.85 ± 0.16 |
| CIFAR-10  | ResNet20| 87.25 ± 0.39  | 6.8 ± 0.8 | 52.47 ± 0.42 |

Simply run ```src/runner.sh``` to generate logs, and then run ```notebooks/tensorboard_log_reader.ipynb``` to average the results of logs. 
