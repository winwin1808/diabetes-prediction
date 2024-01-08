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
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    pregnancies = float(data['pregnancies'])
    glucose = float(data['glucose'])
    bloodPressure = float(data['bloodPressure'])
    skinThickness = float(data['skinThickness'])
    insulin = float(data['insulin'])
    bmi = float(data['bmi'])
    diabetesPedigreeFunction = float(data['diabetesPedigreeFunction'])
    age = float(data['age'])

    input_tensor = torch.tensor([pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]).float()


    with torch.no_grad():
        prediction = model(input_tensor.unsqueeze(0).unsqueeze(0))  # Adjust shape as needed
        predicted_class = prediction.round().item()

    return jsonify({'prediction': predicted_class})


@app.route('/predict_csv', methods=['POST'])
def predict_csv():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})

    file = request.files['file']


    if file.filename == '':
        return jsonify({'error': 'No file selected.'})

    if file and file.filename.endswith('.csv'):

        df = pd.read_csv(file)


        if df.shape[1] != 8:
            return jsonify({'error': 'Invalid CSV format. Expected 8 columns.'})

        input_tensor = torch.tensor(df.values).float()
        with torch.no_grad():
            predictions = model(input_tensor.unsqueeze(1))
            predicted_classes = predictions.round().numpy().tolist()

        return jsonify({'predictions': predicted_classes})

    else:
        return jsonify({'error': 'Invalid file format. Only CSV files are accepted.'})
    
if __name__ == '__main__':
    app.run(debug=True)
