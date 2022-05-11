import base64
from io import BytesIO

import meshio
import pyvista as pv
from flask import Flask, request, render_template
from ipywidgets.embed import embed_snippet

from network import load_stl, load_model, predict

app = Flask(__name__)


def render_colourbar(limits):
    pl = pv.Plotter(window_size=[1024, 256], off_screen=True)

    # Need to add an arbitrary mesh to create the color bar
    plane = [[-5.0, 0.0, 5.0], [-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [5.0, 0.0, 5.0]]
    plane_mesh = pv.PolyData(plane).delaunay_2d()
    pl.add_mesh(plane_mesh, opacity=0.5, clim=limits, scalars=[0, 1, 2, 3])
    pl.clear()  # Discard mesh
    pl.add_scalar_bar(title="Cp", height=0.8, width=0.8, position_x=0.1, position_y=0.1, label_font_size=24, title_font_size=32)
    buf = BytesIO()
    pl.show(screenshot=buf)
    img = base64.b64encode(buf.getbuffer()).decode("ascii")

    return img


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

        return render_template('index.html', result=True, snippet=snippet, colourbar=colourbar)
