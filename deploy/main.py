from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('utils/model.joblib')
vectorizer = joblib.load('utils/vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.get_json()

    # Extract the target and plans from the input data
    target = input_data['target']
    plans = input_data['plans']

    # Construct a DataFrame from the input data
    data = [
        {
            'census_carrier_name': target['carrier'],
            'plan_admin_name': target['name'],
            'carrier_name': plan['carrier'],
            'name': plan['name']
        }
        for plan in plans
    ]
    input_df = pd.DataFrame(data)

    # Preprocess the input data
    input_df['combined_text'] = input_df.apply(lambda x: ' '.join(x.astype(str)), axis=1)
    input_tfidf = vectorizer.transform(input_df['combined_text'])

    # Generate predictions and probabilities
    predictions = model.predict(input_tfidf)
    probabilities = model.predict_proba(input_tfidf)[:, 1]  # Get the probability for the positive class

    # Create a list of plan objects with predictions and probabilities
    predicted_plans = [
        {
            'id': plan['id'],
            'name': plan['name'],
            'carrier': plan['carrier'],
            'prediction': int(prediction),
            'probability': float(probability)
        }
        for plan, prediction, probability in zip(plans, predictions, probabilities)
    ]

    # Sort the predicted plans by prediction and get the top 5
    top_predicted_plans = sorted(predicted_plans, key=lambda x: x['probability'], reverse=True)[:5]

    # Return the top 5 predicted plans as a JSON response
    return jsonify({'predictions': top_predicted_plans})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)