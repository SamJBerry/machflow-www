from flask import Flask, request

from network import load_stl, load_model, predict

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        page = """
            <form action="/" method="POST" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="text" name="aoa">
            <input type="submit">
            </form>
        """
        return page
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file.save(filename)
        stl_file = filename
        data = load_stl(stl_file, int(request.form['aoa']))
        model = load_model()
        pred = predict(model, data)
        cp = pred.cpu().detach().float().numpy()[:, 0]

        return f"{cp}"
