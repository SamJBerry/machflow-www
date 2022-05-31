import base64
import logging
import os
from time import time
from io import BytesIO

import meshio
import pyvista as pv
import werkzeug.exceptions
from flask import Flask, request, render_template
from ipywidgets.embed import embed_snippet
from werkzeug.utils import secure_filename
from network import load_stl, load_model, predict

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'stl'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 10 ** 6  # 10Mb upload limit

# Logging
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# Required for headless image buffering on Linux
try:
    app.logger.info("Starting Xvfb")
    pv.start_xvfb()
except OSError:
    pass


def render_colourbar(limits):
    pl = pv.Plotter(window_size=[1024, 256], off_screen=True)

    # Need to add an arbitrary mesh to create the color bar
    plane = [[-5.0, 0.0, 5.0], [-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [5.0, 0.0, 5.0]]
    plane_mesh = pv.PolyData(plane).delaunay_2d()
    pl.add_mesh(plane_mesh, opacity=0.5, clim=limits, scalars=[0, 1, 2, 3])
    pl.clear()  # Discard mesh
    pl.add_scalar_bar(title="Cp", height=0.8, width=0.8, position_x=0.1, position_y=0.1, label_font_size=24,
                      title_font_size=32)
    buf = BytesIO()
    pl.show(screenshot=buf)
    img = base64.b64encode(buf.getbuffer()).decode("ascii")

    return img


def get_pre_selected():
    return os.listdir("static/pre_selected")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class FileNotAllowed(werkzeug.exceptions.HTTPException):
    code = 406
    description = f"The request file is not in the allowed extension list: {ALLOWED_EXTENSIONS}"


class OOM(werkzeug.exceptions.HTTPException):
    code = 503
    description = "The server is out of memory due to excessive load. Please try again later"


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', result=None, pre_selected=get_pre_selected())
    if request.method == 'POST':
        pre_selected = request.form['pre_file']
        if pre_selected != "0":
            stl_file = "static/pre_selected/" + pre_selected
            app.logger.info(f"Using existing stl: {stl_file}")
        elif allowed_file(request.files['file'].filename):
            file = request.files['file']
            filename = secure_filename(file.filename)
            stl_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(stl_file)
            app.logger.info(f"File uploaded: {file.filename}")
        else:
            raise FileNotAllowed()

        data = load_stl(stl_file, int(request.form['aoa']))
        model = load_model()
        tick = time()
        try:
            pred = predict(model, data)
        except RuntimeError:
            raise OOM()
        tock = time()
        prediction_time = tock - tick
        app.logger.info(f"Prediction made in {prediction_time}")
        cp = pred.cpu().detach().float().numpy()[:, 0]
        mesh = meshio.read(stl_file)
        mesh.point_data['Cp'] = cp
        pl = pv.Plotter(off_screen=True)
        plane = [[-5.0, 0.0, 5.0], [-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [5.0, 0.0, 5.0]]
        plane_mesh = pv.PolyData(plane).delaunay_2d()
        pl.add_mesh(plane_mesh, opacity=0.5)
        pl.add_mesh(mesh)
        pythreejs_renderer = pl.to_pythreejs()
        snippet = embed_snippet([pythreejs_renderer])
        colourbar = render_colourbar([min(cp), max(cp)])

        return render_template('index.html', result=True, snippet=snippet, colourbar=colourbar,
                               pre_selected=get_pre_selected(), prediction_time=prediction_time)


if __name__ == "__main__":
    app.run()
