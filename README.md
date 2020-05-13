

This code also uses code from the following papers

BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning

Bayesian Batch Active Learning as Sparse Subset Approximation

and the repository

https://github.com/kuangliu/pytorch-cifar

Make sure you install all requirements using

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```

The acquisition functions are in multi_bald.py. ICAL is implemented in the function compute_ical and ICAL-pointwise in compute_ical_pointwise
