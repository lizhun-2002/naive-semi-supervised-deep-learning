# [Naive semi-supervised deep learning using pseudo-label](https://doi.org/10.1007/s12083-018-0702-9)
*Original paper: https://rdcu.be/bc9oA*

This is an implementation of naive semi-supervised deep learning on Python 3, Keras, and TensorFlow. 

To facilitate the utilization of large-scale unlabeled data, we propose a simple and effective method for semi-supervised deep learning that improves upon the performance of the deep learning model. 
* First, we train a classifier and use its outputs on unlabeled data as pseudo-labels. 
* Then, we pre-train the deep learning model with the pseudo-labeled data and fine-tune it with the labeled data. 

The repetition of pseudo-labeling, pre-training, and fine-tuning is called **naive semi-supervised deep learning**. 

## Requirements
Python 3.5.5, TensorFlow 1.2.1, and Keras 2.0.6

## Data
We apply this method to the MNIST, CIFAR-10, and IMDB data sets, which are each divided into a small labeled data set and a large unlabeled data set by us. 

## Usage
To run the code, use

```python
python semi_supervised_loop.py
```
