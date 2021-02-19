from flask import Flask, request, jsonify
from PIL import Image
from main_runner import answer_return
import main_runner
import os

app = Flask(__name__)

@app.route("/im_size", methods=["POST"])
def process_image():
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)
    file.save('im-received.jpg')

    #run the model analysis with the image received from the app
    os.system('python my_file.py')
    model_response=main_runner.answer_return()


    return jsonify({'msg': 'success', 'what': model_response})


if __name__ == "__main__":
    app.run(debug=True)