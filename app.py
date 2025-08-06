from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('linear_model.pkl', 'rb'))

@app.route('/')
def home():
    return "House Price Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_features = np.array(data['input']).reshape(1, -1)
    prediction = model.predict(input_features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
