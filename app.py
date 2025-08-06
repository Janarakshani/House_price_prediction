from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return "House Price Predictor is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([list(data.values())])
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
