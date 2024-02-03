# Toxicity classification

## Prerequisites

This example was created using the following prerequisites:
- Windows OS
- Python 3.9
- Virtual environment

It should be possible to run it on other operating systems that support Python.
The script will most probably work on any Python 3 installation. Here Python 3.9 was used.

A virtual environment is optional. Using a virtual environment makes it easier to create a environment with just those modules you need. This is separate from what modules are installed in the Python version that is installed on the operating system.

If you don't want to use a virtual environment you can use pip to install the modules that are listed
in the requirements.txt file.
Using this command pip will install the modules:
'pip install --requirement requirements.txt'

### Create the virtual environment

This batch file is Windows specific
- Open a terminal window
- Run: './create_venv.bat'

## Data

The tocix tweets dataset as published here was used to train the model:
https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset

The dataset contains just two fields:
- text: a text extracted from Twitter.
- is_toxic: a label to classify whether the text is considered to be toxic or not.

Copy the dataset CSV file in the root directory of this project and make sure its name is 'toxicity_en.csv'

## Train neural network

First a neural network needs to be trained in the input data.
- Run: 'python train_neural_network.py'

## Run the toxicity classification GUI

Now that the model has been trained the GUI application can be started to enter a sentence
and make the model predict whether this sentence is considered toxic or not.
- Run: 'python predict_toxicity_gui.py'
