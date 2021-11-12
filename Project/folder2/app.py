from flask import Flask
from flask import render_template, request
from ge_ml import main

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():

    return render_template('index.html', title='Welcome')


main('Soul_rune')
# main('Soul_rune')
# main('Soul_rune')

# @app.route('/rune', methods=['POST'])
# def rune():
#     if request.method == 'POST':
#         test_rune = request.form['rune']
#         print(request.form['rune'])
#         main(test_rune)
#         return 'ML done'


if __name__ == '__main__':
    app.run(debug=True)
