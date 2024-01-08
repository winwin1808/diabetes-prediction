from flask import Flask, request, jsonify, render_template
import torch
from model import ComplexDiabetesLSTM  # Import your model class
import pandas as pd
import openpyxl  

app = Flask(__name__)
app.static_folder = 'static'
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 

# Load your trained model
input_size = 8
model = ComplexDiabetesLSTM(input_size)
model.load_state_dict(torch.load('diabetes_model.pth', map_location='cpu'))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' is in the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    # Extract data from form fields
    pregnancies = float(data['pregnancies'])
    glucose = float(data['glucose'])
    bloodPressure = float(data['bloodPressure'])
    skinThickness = float(data['skinThickness'])
    insulin = float(data['insulin'])
    bmi = float(data['bmi'])
    diabetesPedigreeFunction = float(data['diabetesPedigreeFunction'])
    age = float(data['age'])

    # Create input tensor for the model
    input_tensor = torch.tensor([pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]).float()

    # Get prediction from the model
    with torch.no_grad():
        prediction = model(input_tensor.unsqueeze(0).unsqueeze(0))  # Adjust shape as needed
        predicted_class = prediction.round().item()

    # Return the result as JSON
    return jsonify({'prediction': predicted_class})


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})

    file = request.files['file']

    # If the file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected.'})

    if file and file.filename.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(file)

        # Check if the CSV has the right number of columns (8)
        if df.shape[1] != 8:
            return jsonify({'error': 'Invalid CSV format. Expected 8 columns.'})

        # Process CSV data and make prediction
        input_tensor = torch.tensor(df.values).float()
        with torch.no_grad():
            predictions = model(input_tensor.unsqueeze(1))  # Adjust shape as needed
            predicted_classes = predictions.round().numpy().tolist()

        return jsonify({'predictions': predicted_classes})

    else:
        return jsonify({'error': 'Invalid file format. Only CSV files are accepted.'})
    
if __name__ == '__main__':
    app.run(debug=True)
