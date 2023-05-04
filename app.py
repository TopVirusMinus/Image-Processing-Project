from flask import Flask, render_template, request
import base64
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def filter1(image):
    # Apply filter 1 processing here
    return image

def filter2(image):
    # Apply filter 2 processing here
    return image

def filter3(image):
    # Apply filter 3 processing here
    return image

@app.route('/process-image', methods=['POST'])
def process_image():
    image_data = request.form.get('image_data')
    filter_type = request.form.get('filter_type')

    image_data = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if filter_type == 'filter1':
        processed_image = filter1(image)
    elif filter_type == 'filter2':
        processed_image = filter2(image)
    elif filter_type == 'filter3':
        processed_image = filter3(image)

    retval, buffer = cv2.imencode('.jpg', processed_image)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
    return {'processed_image': f'data:image/jpeg;base64,{processed_image_base64}'}

if __name__ == '__main__':
    app.run(debug=True)
