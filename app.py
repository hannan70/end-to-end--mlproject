from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import numpy as np

application = Flask(__name__)

app = application

@app.route("/")
def index_page():
    return render_template('index.html') 


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df=data.get_data_as_data_frame()

        predict_pipeline=PredictPipeline()
 
        results=predict_pipeline.predict_data(pred_df)

        formated_result = f"{results[0]:.2f}"
  
        return render_template('index.html',results=formated_result)


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)   

