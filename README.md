# MemberInference-by-LossThreshold

This is a minimal implementation of [loss thresholding attack](https://arxiv.org/abs/1709.01604) to infer membership status (i.e., part of training data or not) of data samples.
Briefly, we have a threshold **T** (e.g., avg. training loss), and we predict a sample 'member' if sample loss < **T**.


Attack code in available in ```src/attacks.py``` and some results on common image datasets (averaged over 5 runs) are as follows.

| Dataset | Test Acc (%) | Generalization Gap (%) |  MIA Balanced Acc. (%)
| ------------- | ------------- | ------------- | ------------- | 
| MNIST  | 99.2 ± 0.05 | 0.76 ± 0.04 | 50.34% ± 0.1 |
| Fashion-MNIST | 90.77 ± 0.11  | 5.02 ± 0.63 | 51.85 ± 0.16 |
| CIFAR-10  | 87.25 ± 0.39  | 6.8 ± 0.8 | 52.47 ± 0.42 |

Simply run ```src/runner.sh``` to generate logs, and then run ```notebooks/tensorboard_log_reader.ipynb``` to average the results of logs. 
