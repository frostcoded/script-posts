from flask import Flask, request, jsonify
from nltk.classify import NaiveBayesClassifier
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, filename='app.log')
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

# Firebase Setup
cred = credentials.Certificate(os.getenv("FIREBASE_KEY_PATH"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Training Storage
mood_training_data = []
tonality_training_data = []
mood_classifier = None
tonality_classifier = None

def word_feats(text):
    """Convert text to feature set for classification."""
    return {word: True for word in text.lower().split()}

def load_training_from_firebase():
    """Load and process training data from Firestore."""
    mood16
    mood_training_data.clear()
    tonality_training_data.clear()

    try:
        docs = db.collection("messages").stream()
        for doc in docs:
            data = doc.to_dict()
            message = data.get("message", "").strip()
            mood_id = data.get("mood_id")
            tonality_id = data.get("tonality_id")
            weight = float(data.get("weight", 1.0))

            if not message or not mood_id or not tonality_id:
                logger.warning(f"Skipping invalid document: {doc.id}")
                continue

            repeats = max(1, int(round(weight)))
            features = word_feats(message)
            mood_training_data.extend([(features, mood_id)] * repeats)
            tonality_training_data.extend([(features, tonality_id)] * repeats)
    except Exception as e:
        logger.error(f"Firestore error: {str(e)}")
        raise

@app.route('/retrain', methods=['POST'])
def retrain_classifiers():
    """Retrain classifiers with Firestore data."""
    global mood_classifier, tonality_classifier
    try:
        load_training_from_firebase()
        if not mood_training_data or not tonality_training_data:
            return jsonify({"error": "No valid training data in Firestore"}), 400

        mood_classifier = NaiveBayesClassifier.train(mood_training_data)
        tonality_classifier = NaiveBayesClassifier.train(tonality_training_data)
        logger.info(f"Classifiers retrained: {len(mood_training_data)} mood examples, {len(tonality_training_data)} tonality examples")
        return jsonify({
            "status": "Classifiers retrained successfully",
            "mood_examples": len(mood_training_data),
            "tonality_examples": len(tonality_training_data)
        })
    except Exception as e:
        logger.error(f"Retrain error: {str(e)}")
        return jsonify({"error": "Failed to retrain classifiers"}), 500

@app.route('/classify', methods=['POST'])
def classify_message():
    """Classify a message for mood and tonality."""
    if mood_classifier is None or tonality_classifier is None:
        return jsonify({"error": "Classifiers not trained. Call /retrain first."}), 400

    data = request.get_json()
    message = data.get("message", "").strip()
    if not message or len(message) > 1000:
        return jsonify({"error": "Message is required and must be under 1000 characters"}), 400

    features = word_feats(message)
    mood_result = mood_classifier.classify(features)
    tonality_result = tonality_classifier.classify(features)
    mood_prob = mood_classifier.prob_classify(features).prob(mood_result)

    return jsonify({
        "message": message,
        "mood_prediction": mood_result,
        "mood_confidence": round(mood_prob, 2),
        "tonality_prediction": tonality_result
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv("FLASK_PORT", 5000)))
