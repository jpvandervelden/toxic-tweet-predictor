# These imports are necessary to run the script
import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load pre-trained spaCy model with word embeddings (e.g., en_core_web_md)
# en_core_web_md is a medium-sized model trained on web text for English language.
nlp = spacy.load('en_core_web_md')

def preprocess_text(text):
    """
    Preprocess each sentence by tokenizing and vectorizing it using spaCy.
    The endresult is that each sentence is represented as a 300-dimensional vector.
    Which means the sentence is represented by 300 numerical values.
    An example of what a vector might look like is as follows:
    [-1.4259182   1.6142589  -2.3934095  -0.53022885  3.952562    1.1934577
     0.5510842   4.4830103  -0.48629308 -0.33751875  6.460623    1.3925962
     ...
     -3.1235964   0.40293008  1.604494    1.8825852   1.0050238  -0.8628506
     -1.3323885  -1.2543931   1.4474171  -0.6424192  -0.925169   -1.1854464]
    """
    doc = nlp(text)
    # Use the vector representation of the entire document
    return doc.vector

def create_embeddings(csv_file_path, output_csv_path):
    """
    Create vector embeddings for sentences in a CSV file and save the results to a new CSV file.
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Preprocess text and create embeddings and add them to the DataFrame
    df['embedding'] = df['text'].apply(preprocess_text)

    # Save the DataFrame with embeddings to a new CSV file
    df.to_csv(output_csv_path, index=False)

    # Return the DataFrame
    return df

# Here's where the script starts execution

# Define the input and output file paths
input_csv_file = 'toxicity_en.csv'
output_csv_file = 'output_embeddings.csv'

# Create embeddings for the toxicity dataset
# Embeddings are numerical representations of the sentences.
# Numerical values are required for neural networks to process the data as they can't process text directly.
df = create_embeddings(input_csv_file, output_csv_file)

# The label in a neural network is the value we want to predict
# Also for the label field the neural network requires numerical input
# So we need to encode the 'is_toxic' label to numerical values
# Encode the 'is_toxic' labels to numerical values (0 for 'Not toxic', 1 for 'Toxic')
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['is_toxic'])

# To be able to validate how well the model can predict toxicity 
# we need to split the data into training and testing sets
# The training part will be used to train the model
# And the testing part will be used to evaluate the model on data the trained model has never seen
X_train, X_test, y_train, y_test = train_test_split(df['embedding'].tolist(), df['label'].tolist(),
                                                    test_size=0.2, random_state=42)

# Convert lists to numpy arrays as the neural network works with numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Build a simple neural network model
# The model will have one input layer, one hidden layer and one output layer
# Sequential is a linear stack of layers
model = Sequential()
# The input layer has 300 neurons as the input data has 300 dimensions
# The hidden layer has 64 neurons and uses the ReLU activation function
# The Spacy embeddings use a dimensionality of 300 so we need to specify that as the input shape
model.add(Dense(64, activation='relu', input_shape=(300,)))
# The output layer has only one neuron and uses the sigmoid activation function
# The sigmoid activation function is used as the output is binary (0 or 1)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
# Adam is an optimization algorithm used to update network weights iteratively based on training data
# The loss function is binary_crossentropy as the output is binary
# The metric we want to optimize is accuracy
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# Epochs is the number of times the model will see the entire training set
# Batch size is the number of samples the model will see before updating the weights
# Validation split is the percentage of the training set that will be used for validation
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
# This is data the model has never seen before
predictions = model.predict(X_test)
# Round the predictions to 0 or 1
rounded_predictions = np.round(predictions)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, rounded_predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the trained model for later use.
model.save('toxicity_model.h5')