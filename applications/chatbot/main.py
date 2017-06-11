# -*- coding: utf-8 -*-

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import os

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('chat.html')


@app.route("/ask", methods=['POST'])
def ask():
    message = str(request.form['messageText'])

    while True:
        if message == "quit":
            exit()
        else:
            # print bot_response
            return jsonify({'status': 'OK', 'answer': "hah"})


if __name__ == "__main__":
    app.run(debug=True)
