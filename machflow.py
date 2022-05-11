import meshio
import pyvista as pv
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
        mesh = meshio.read(stl_file)
        mesh.point_data['Cp'] = cp
        pl = pv.Plotter(off_screen=True)
        plane = [[-5.0, 0.0, 5.0], [-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [5.0, 0.0, 5.0]]
        plane_mesh = pv.PolyData(plane).delaunay_2d()
        pl.add_mesh(plane_mesh, opacity=0.5)
        pl.add_mesh(mesh)
        pl.export_html('pyvista.html')
        f = open('pyvista.html', "r")
        lines = f.read()
        f.close()

        return lines
