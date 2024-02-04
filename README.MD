# Toxicity classification

## Prerequisites

This example was created using the following prerequisites:
- Python 3.8.10

It should be possible to run it on any operating systems that support Python.
The script will most probably work on any Python 3 installation.

### 3rd party modules

You can use pip to install the third party modules that are listed in the requirements.txt file. Pip is the package manager for Python which can install third party modules. Pip is by default part of the Python installation.

Open a terminal window.
Make sure you are in the directory where the requirements.txt file is.
Using this command pip will install the modules:
'pip install --requirement requirements.txt'
This will take some time to finish.

### Spacy language model

As this solution uses a language model to create embeddings a prebuilt language model is used created by Spacy. This model needs to be installed.
Run 'python -m spacy download en_core_web_md'

## Architecture of the solution

In order to create an application that can take a sentence as input and return whether the sentence is toxic or not we need a number of components.

### Data preparation
The Kaggle dataset is a curated dataset which has a high data quality. There are no missing values that need imputing, nor does data need to be standardized. So no work to be done here.

### Sentence Embeddings
We need to train a neural network on the input data set. Neural networks work using numerical data and the input data set is text based. We first need to find a numerical way to represent the text.
The solution chosen here is to use language embeddings. Language embeddings create a vector for a piece of text. The semantics of the text is captured in the vector. In this case vectors representing toxic comments will be closer to each other than to non-toxic comments.

### Train a Neural network
With the input in the form of language embeddings a neural network can be trained. The input contains the label field which indicates whether the sentence at hand is considered toxic or not. With this supervised learning can be applied and a model can be trained which predicts toxicity of sentences.

### Application
To allow end users to enter their own example sentences and get a prediction an application is created which lets the user enter a sentence and click a button to run it against the model and get back a prediction. The prediction is shown to the user. A prediction lower than 0.5 is considered non-toxic, anything above 0.5 is considered toxic. 0 and 1 and the extremes, so 0 is absolutely non-toxic and 1 is extremely toxic.

As the model has an accuracy of 80-90%, it can happen that new input is wrongly predicted. Give it a try to see how well it works.

## Training and executing the application

The following sections describe the steps that need to be taken in order the train the neural network and to run the application.

### Data

The tocix tweets dataset as published here is used to train the model:
https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset

The dataset contains just two fields:
- text: a text extracted from Twitter.
- is_toxic: a label to classify whether the text is considered to be toxic or not.

Download the file from Kaggle. Copy the dataset CSV file in the root directory of this project and make sure its name is 'toxicity_en.csv'

## Train neural network

First a neural network needs to be trained in the input data.
Run: 'python train_neural_network.py'
When I ran it the model finished with accuracies between 80 and 90%.

## Run the toxicity classification GUI

Now that the model has been trained the GUI application can be started to enter a sentence and make the model predict whether this sentence is considered toxic or not.
Run: 'python predict_toxicity_gui.py'

