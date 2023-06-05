from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import pprint

app = Flask(__name__, template_folder='templates', static_folder='static')


Pkl_Filename = "rf_tuned.pkl"
with open(Pkl_Filename, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        features = [int(x) for x in request.form.values()]
        pprint.pprint(features)
        final = np.array(features).reshape((1, 6))
        pprint.pprint(final)
        pred = model.predict(final)[0]
        pprint.pprint(pred)

        if pred < 0:
            return render_template('op.html', pred='Error calculating Amount!')
        else:
            return render_template('op.html', pred='Expected amount is {0:.3f}'.format(pred))
    except Exception as e:
        return render_template('op.html', pred='An error occurred: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
