
import os

import numpy as np
from flask import Flask, request
from flask_cors import CORS
# import tensorflow as tf
app = Flask(__name__)
CORS(app)

@app.route('/hello', methods=['GET'])
def hello():
    return "Hello World"

if __name__ == '__main__':
    app.run(debug=True)