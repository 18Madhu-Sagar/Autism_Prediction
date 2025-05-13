from flask import Flask, request, render_template
import pickle
import numpy as np

model = pickle.load(open('autism.pkl', 'rb'))

app = Flask(__name__)

@app.route('/home')
def home():
    return render_template('main.html')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/help')
def helps():
    return render_template('help.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        d = request.form['Ch']
        data = {}
        for key in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10','q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20','q21', 'q22', 'q23', 'q24', 'q25']:
            data[key] = 0 if request.form[key] == 'yes' else 1
        qch=sum(data.values())
        data['age'] = int(request.form['age'])
        data['qch'] = qch
        data['ethnicity'] = int(request.form['ethnicity'])
        data['jaundice'] = 1 if request.form['Jau'] == 'yes' else 0
        data['fma'] = 1 if request.form['fma'] == 'yes' else 0
        
        
           # Assuming age is one of the features
        
        result = np.array([list(data.values()) ])
        print(data.keys())
        prediction = model.predict(result)
        if prediction[0] == 1:
            data['qch']=data['qch']*0.3
            if data['ethnicity']==5:
                data['qch']+=0.08
            if data['ethnicity']==0 or data['ethnicity']==1:
                data['qch']+=0.07
            if data['ethnicity'] == 7:
                data['qch']+=0.05
            if data['ethnicity']==6:
                data['qch']+=0.03
            if data['ethnicity']==2 or data['ethnicity']==4 or data['ethnicity']==8 or data['ethnicity']==9 or data['ethnicity']==10:
                data['qch']+=0.02
            if data['jaundice']==1:
                data['qch']+=0.085
            if data['jaundice']==1:
                data['qch']+=0.08
        sol=prediction[0]
        data['qch']*=10
        
        return render_template('dr.html', prediction=sol,text=d,re=int(data['qch']))
        
    except Exception as e:
        return render_template('error.html', error="please fill all inputs currectly")
import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)