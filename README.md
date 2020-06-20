# memoCNN: A convolutional neural network to classify metamodels

This repository contains the source code implementation of memoCNN and the datasets used to replicate the experimental results for the following paper: 

_Convolutional neural networks for enhanced classification mechanisms of metamodel repositories_

The paper has been submitted to Journal of Systems and Software and it is now under review.

## Introduction

The classification of metamodels into independent categories helps developers quickly approach the most relevant ones to their needs by narrowing down the search scope. However, the manual annotation of metamodels is both time-consuming and inapplicable to a large amount of data. In this sense, it is essential to have decent machinery to automatize such a process. We propose memoCNN as a novel approach to classify metamodels. We first devised a method to transform metamodels into a processable format, and built a deep neural network to classify them afterwards. Experimental results show that the approach effectively categorizes input data, and outperforms a state-of-the-art baseline.

<p align="center">
<img src="https://github.com/MDEGroup/memoCNN/blob/master/images/memoCNN.png" width="600">
</p>


The proposed architecture allows for both offline and online training. By the former, one can train the network on a PC or a laptop installed with TensorFlow and Keras. By the latter, the code is executed directly on Google Colaboratory 1 connecting to metadata stored into Google Drive. The final outcome of the training process is a set of weights and biases, which then can be used to deploy the network. An example of a memoCNN network is shown below.

<p align="center">
<img src="https://github.com/MDEGroup/memoCNN/blob/master/images/CNN.png" width="750">
</p>


## Repository Structure

This repository is organized as follows:

* The folder [tool](./tool) contains the implementation of the memoCNN tool we developed:
	* ***.ipynb** files contain the notebooks for running memoCNN.
	* **data_helpers_2.py**	is the library used to read external data files stored in Google Drive.
* The folder [dataset](./dataset) contains the datasets described in the paper used to evaluate memoCNN:	
	* [unigram](./dataset/unigram): metadata generated using uni-gram.
	* [bigram](./dataset/bigram): metadata generated using bi-gram.
	* [ngram](./dataset/ngram): metadata generated using n-gram.
	* [HoldOut](./dataset/HoldOut): the metadata needed for RQ1 in the paper, corresponding to various hold-out settings.

* In each of the dataset folder mentioned above, you will find ten sub-folders, i.e., **Roundi**, each corresponds to one fold of training (train.csv) and testing data (test.csv).

## How to execute the code

File **memoCNNi.ipynb** is the source code file corresponding to configuration C<sub>i</sub> in the paper. For instance, **memoCNN1.ipynb** represents configuration C<sub>1</sub> with uni-gram as the encoding scheme, input size 94 x 94, convolutional kernel 3 x 3 x 1 x 16.

The experimental configurations are as follows:

<p align="center">
<img src="https://github.com/MDEGroup/memoCNN/blob/master/images/Configurations.png" width="800">
</p>


You need to upload all the metadata in the [dataset](./dataset) directory as well as the [tool](./tool) directory to your Google Drive account. Please follow the instructions below:

* In your Google Drive account, create a folder with the name **Colab Notebooks**.


* Upload all the sub-folders in the [dataset](./dataset) folder to **Colab Notebooks**.


* Upload all the notebooks within the [tool](./tool) folder, i.e., ***.ipynb** files and **data_helpers_2.py** to **Colab Notebooks**.


After that, you can run the notebooks on Google Colab. This [post](https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c) provides a detailed introduction on how to work with Google Colab.


If you run for the first time, Google Colab will ask you to authenticate your Google Drive as shown in the figure below:

<p align="center">
<img src="https://github.com/MDEGroup/memoCNN/blob/master/images/GoogleColabAuth.png" width="800">
</p>

You have to click the link above, and you will be redirected to the next page to get the access token:

<p align="center">
<img src="https://github.com/MDEGroup/memoCNN/blob/master/images/GoogleColabAuth5.png" width="500">
</p>

And finally paste the token to the box and press Enter to run the notebook:

<p align="center">
<img src="https://github.com/MDEGroup/memoCNN/blob/master/images/GoogleColabAuth3.png" width="800">
</p>


## Results

Once Google Colab finishes its computation, it will display the final result that looks like as in the picture below

<p align="center">
<img src="https://github.com/MDEGroup/memoCNN/blob/master/images/Result.png" width="500">
</p>




