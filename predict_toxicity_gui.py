# These imports are necessary to run the script
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from tensorflow.keras.models import load_model
import numpy as np
import spacy

class ToxicityPredictionApp(QWidget):
    '''
    This class creates a simple GUI for predicting toxicity of a sentence.
    '''

    def __init__(self):
        '''
        Initialize the GUI and load the trained model.
        '''

        super().__init__()

        # Load the trained model
        self.trained_model = load_model('toxicity_model.h5')

        # Load spaCy model
        self.nlp = spacy.load('en_core_web_md')

        # Initialize the GUI
        self.init_ui()

    def init_ui(self):
        ''' 
        Initialize the GUI components.
        '''	

        self.setWindowTitle('Toxicity Prediction')

        # Create GUI components
        # Label and input field for the user to enter a sentence
        self.label = QLabel('Enter a sentence:')
        self.entry = QLineEdit()
        # Button to predict the toxicity of the entered sentence
        self.predict_button = QPushButton('Predict')
        # Label to display the prediction result
        self.result_label = QLabel('Prediction: ')

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.entry)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        # Connect button click event to the prediction function
        self.predict_button.clicked.connect(self.on_predict_button_click)

    def preprocess_text_for_prediction(self, text):
        '''
        Preprocess the input text for prediction.
        '''

        # Tokenize and vectorize the input text using spaCy in the same way as
        # this was done during training the model, otherwise the model will not be able to make predictions.
        doc = self.nlp(text)
        return doc.vector

    def predict_toxicity(self, sentence):
        '''
        Predict the toxicity of a sentence. 
        '''

        # Preprocess the input sentence
        preprocessed_sentence = self.preprocess_text_for_prediction(sentence)
        # Convert the preprocessed sentence to a numpy array
        input_data = np.array([preprocessed_sentence])
        # Make a prediction
        prediction = self.trained_model.predict(input_data)
        return prediction[0, 0]

    def on_predict_button_click(self):
        '''
        Handle the button click event.
        '''

        # Get the user input
        user_input = self.entry.text()

        # Check if the user entered a sentence
        if not user_input:
            self.result_label.setText("Prediction: Please enter a sentence.")
            return

        # Predict the toxicity of the entered sentence
        toxicity_probability = self.predict_toxicity(user_input)

        # Convert probability to a binary label
        # The probability is a floating point number between 0 and 1
        # If the probability is greater than or equal to 0.5, the sentence is considered toxic
        threshold = 0.5
        predicted_label = 'Toxic' if toxicity_probability >= threshold else 'Not Toxic'

        # Display the prediction result
        self.result_label.setText(f"Prediction: {predicted_label} (Probability: {toxicity_probability:.2f})")


# Run the application
if __name__ == '__main__':
    # Create the application
    app = QApplication(sys.argv)
    # Create the main window
    main_app = ToxicityPredictionApp()
    # Show the main window
    main_app.show()
    sys.exit(app.exec_())
