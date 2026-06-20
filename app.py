from flask import Flask, render_template, request
import statsmodels.api as sm
import pandas as pd
import numpy as np

app = Flask(__name__)
model = sm.load('Batch_model_new.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            hp = float(request.form.get('hp'))
            sp = float(request.form.get('sp'))
            wt = float(request.form.get('wt'))

            input_df = pd.DataFrame({
                'LOG_HP': [np.log(hp)],
                'SP': [sp],
                'WT': [wt]
            })

            result = model.predict(input_df)
            prediction = round(float(result[0]), 4)
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
