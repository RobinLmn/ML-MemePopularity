import flask
import model as m
import processdata as data
import numpy as np
import PIL.Image
import os
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def to_string(c):
    if c == 0:
        return "1/4: Boriiiing."
    elif c == 1:
        return "2/4: It is merely entertaining."
    elif c == 2:
        return "3/4: Pretty funny."
    elif c == 3:
        return "4/4: Oooooh that is dank."

def predict(file):
    image = ""

    try:
        img = PIL.Image.open(file)
        image = data.process_image(img)
        img.close()
    except FileNotFoundError:
        img = data.url_to_image(file)
        image = data.process_image(img)
        img.close()

    model = m.get()
    result = model.predict_classes(np.array( [image,] ) )
    return to_string(result[0])

default = "https://i.redd.it/65bzzioisir01.jpg"

@app.route("/", methods=['GET', 'POST'])
def main():
    image = ""
    if flask.request.method == 'POST':
        if 'file' not in flask.request.files:
            image = default
        file = flask.request.files['file']
        if file.filename == '':
            image = default
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    else:
        image = default
    result = predict(image)
    return flask.render_template("index.html", result=result, image=image)


app.run(host='0.0.0.0', port=50000)
