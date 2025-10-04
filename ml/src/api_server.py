from flask import Flask, request, jsonify
from hybrid_model import hybrid_predict

app = Flask(__name__)

@app.route('/scan', methods=['POST'])
def scan_file():
    data = request.json
    features = data.get("features")
    if not features:
        return jsonify({"error": "No features provided"}), 400
    label, score = hybrid_predict(features)
    return jsonify({"result": label, "confidence": round(score, 3)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)