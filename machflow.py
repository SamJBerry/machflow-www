from flask import Flask, request, render_template

from network import load_stl, load_model, predict

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', result=None)
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file.save(filename)
        stl_file = filename
        data = load_stl(stl_file, int(request.form['aoa']))
        model = load_model()
        pred = predict(model, data)
        cp = pred.cpu().detach().float().numpy()[:, 0]

        return render_template('index.html', result=cp)
