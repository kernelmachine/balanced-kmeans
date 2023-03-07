# Balanced K-Means using PyTorch

PyTorch implementation of balanced kmeans. Based on https://github.com/subhadarship/kmeans_pytorch

# Requirements
* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.6

# Installing from source

To install from source and develop locally:
```
git clone https://github.com/kernelmachine/balanced-kmeans/
cd balanced-kmeans
pip install --editable .
```

Install pytorch 

```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Install additional dependencies

```
pip install matplotlib tqdm scikit-learn numba
```

# Run example

This will output a plot of clusters in a pdf file.

```
python cluster.py
```

You can check out the notebook `example.ipynb` as well.