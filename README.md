Active learning by loss
=== 

### Problem definition
To see if the active learning method works well.
- dataset -> CIFAR10
- network -> ResNet18

### Result
![result](/figures/result.png)
- The result of active learning(10000 samples)is almost identical to the result of training the model on the full data(50000samples)  
- The model trained on random samples(10000 samples) have low accuracy

### Reference
[1] Yoo, Donggeun, and In So Kweon. "Learning loss for active learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.