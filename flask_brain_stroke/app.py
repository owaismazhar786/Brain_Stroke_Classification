from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    features = [float(x) for x in request.form.values()]

    # Make a prediction using the loaded model
    prediction = model.predict([features])

    # Map the prediction to the corresponding class label
    if prediction[0] == 0:
        result = "Unlikely to have a stroke"
    else:
        result = "Likely to have a stroke"

    # Render the predict.html template with the prediction result
    return render_template('predict.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
