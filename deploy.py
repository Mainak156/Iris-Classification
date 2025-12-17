# ...existing code...
import os
import logging
from flask import Flask, render_template, request
import pickle

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.sav")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_model(path):
    if not os.path.exists(path):
        logger.error("Model file not found at: %s", path)
        return None
    try:
        with open(path, "rb") as f:
            m = pickle.load(f)
            logger.info("Model loaded from %s", path)
            return m
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        return None

model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', result="Error: model not available")

    try:
        sl = request.form.get('sepal_length', '').strip()
        sw = request.form.get('sepal_width', '').strip()
        pl = request.form.get('petal_length', '').strip()
        pw = request.form.get('petal_width', '').strip()

        if not all([sl, sw, pl, pw]):
            return render_template('index.html', result="Error: all measurements are required")

        sepal_length = float(sl)
        sepal_width  = float(sw)
        petal_length = float(pl)
        petal_width  = float(pw)
    except ValueError:
        return render_template('index.html', result="Error: invalid numeric input")

    # basic range validation (helps avoid accidental bad values)
    if not (0 <= sepal_length <= 30 and 0 <= sepal_width <= 30 and 0 <= petal_length <= 30 and 0 <= petal_width <= 30):
        return render_template('index.html', result="Error: input values out of expected range")

    try:
        pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        # ensure string for template checks
        pred_label = str(pred)
        # optional: compute confidence if model supports it
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba([[sepal_length, sepal_width, petal_length, petal_width]])[0]
                confidence = max(probs) * 100.0
            except Exception:
                confidence = None

        # pass confidence as well (template can be extended to show it)
        return render_template('index.html', result=pred_label, confidence=confidence)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return render_template('index.html', result="Error: prediction failed")

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)