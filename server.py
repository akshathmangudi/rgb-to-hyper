# code has to be tested properly once the both model's run , the code is incomplete
# -------- Code for laptop -------- #
from flask import Flask, request, Response, jsonify
import os
from PIL import Image


app = Flask(__name__)

# Path to save the image
folder = 'uploads'
if not os.path.exists(folder):
    os.makedirs(folder)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return Response(
            response=jsonify({'error': 'No file part'}).get_data(as_text=True),
            status=400,
            mimetype='application/json'
        )

    file = request.files['image']
    if file.filename == '':
        return Response(
            response=jsonify({'error': 'No selected file'}).get_data(as_text=True),
            status=400,
            mimetype='application/json'
        )


    file_path = os.path.join(folder, file.filename)
    file.save(file_path)


    result = run_model(file_path)  # Model results , yet to be implemented


    return Response(
        response=jsonify({'result': f'Predicted class: {result}'}).get_data(as_text=True),
        status=200,
        mimetype='application/json'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# -------- Code for Pi -------- #
import requests
from picamera import PiCamera
from time import sleep

# Laptop Link
url = ''


def capture_and_send_image():
    camera = PiCamera()
    camera.start_preview()
    sleep(2)
    camera.capture('image.jpg')
    camera.stop_preview()

    with open('image.jpg', 'rb') as img:
        files = {'image': img}
        try:

            response = requests.post(url, files=files)
            if response.status_code == 200:
                final = response.json()
                print("Success:", final['result'])
            elif response.status_code == 400:
                print("Error:", response.json())
            else:
                print(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"failed: {e}")


if __name__ == "__main__":
    capture_and_send_image()

