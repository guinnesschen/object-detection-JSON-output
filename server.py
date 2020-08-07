from flask import Flask, request
from process_image import process

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def process_image():
    return process(request.get_json()["data"])

if __name__ == "__main__":
    app.run(host="0.0.0.0")
