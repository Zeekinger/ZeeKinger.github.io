from flask import Flask, request, render_template,jsonify

app = Flask(__name__)
port = 80
@app.route('/')
@app.route('/home')
def home():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True,port=port)