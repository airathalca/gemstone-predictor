from src.gemstonePrediction.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

prediction_pipeline = PredictionPipeline()

@app.route('/')
@cross_origin()
def home():
    return render_template("index.html")

# Route for form page
@app.route('/form')
@cross_origin()
def form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict_datapoint():
  try:
    # Get the form data
    data = CustomData(
        carat=float(request.form.get('carat')),
        depth=float(request.form.get('depth')),
        table=float(request.form.get('table')),
        x=float(request.form.get('x')),
        y=float(request.form.get('y')),
        z=float(request.form.get('z')),
        cut=request.form.get('cut'),
        color=request.form.get('color'),
        clarity=request.form.get('clarity')
    )
    
    # Process the data
    final_data = data.get_data_as_df()
    pred = prediction_pipeline.predict(final_data)
    
    # Round the result to 2 decimal places
    result = "{:.2f}".format(float(pred[0]))
    # Return the result as JSON
    return jsonify({'prediction': result})

  except Exception as e:
    return jsonify({'error': str(e)})

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)