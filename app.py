import os
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = 'trained_model.pkl'
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"{model_path} is missing!")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Model is loaded successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([list(data.values())])
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
