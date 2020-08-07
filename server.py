from flask import Flask, request
from main import main

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def process_image():
    return main(request.get_json()["data"])

if __name__ == "__main__":
    app.run()
