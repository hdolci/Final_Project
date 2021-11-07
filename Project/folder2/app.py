from flask import Flask
from flask import render_template
# import ge_ml

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    name = 'HD'
    return render_template('index.html', title='Welcome', username=name)

# @app.route("/rune-price")
# def get-price():

#     return
