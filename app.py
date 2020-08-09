import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_breast_cancer_detector.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['clump_thickness', 'cell_size', 'cell_shape', 'marginal_adhesion', 
	'single_epithelial_size', 'bare_nuclei','bland_chromatin','normal_nucleoli']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 0:
        res_val = "** Breast Cancer Predicted**"
    else:
        res_val = " No Breast Cancer"
        

    return render_template('index.html', prediction_text='ML algo says{}'.format(res_val))

if __name__ == "__main__":
    app.run()
