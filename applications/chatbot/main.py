# -*- coding: utf-8 -*-

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

from applications.chatbot.model import Seq2Seq

app = Flask(__name__)
model = Seq2Seq()


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
            return jsonify({'status': 'OK',
                            'answer': model.utter(message)})


if __name__ == "__main__":
    app.run(debug=True)


