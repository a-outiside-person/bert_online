from flask import Flask, render_template, request, jsonify
from predict import Predictor

app = Flask(__name__)
predictor = Predictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    title = data.get('title', '')
    if not title:
        return jsonify({'error': 'Title is empty'}), 400
    
    result = predictor.predict(title)
    print('title:', title)
    print('result:', result)
    return jsonify({'category': result})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
