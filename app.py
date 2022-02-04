from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import matplotlib.pyplot as plt
from keras.preprocessing import image
import os
from cv2 import VideoCapture,imwrite

import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("my_model_SGD_4_Epochs")

temp_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
             'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
             'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17,
             'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
             'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}
class_dict = {i[1]:i[0] for i in temp_dict.items()}


img = ''
result = ''
def trigger_webcam():
    camera = VideoCapture(0)
    return_value, image = camera.read()
    del(camera)
    return image

app = Flask(__name__)


@app.route('/upload')
def upload():
    return """
<html>
   <body>
      <form action = "http://localhost:5000/uploader" method = "POST" 
         enctype = "multipart/form-data">
         <input type = "file" name = "file" />
         <input type = "submit"/>
      </form>
   </body>
</html>
"""

@app.route('/')
def home():
    try:
        os.remove("/home/vb/Documents/Aegis/Deep Learning/Project/static/image_01.png")
    except:
        None
    return render_template('capture_image.html')

@app.route('/live_capture', methods = ['GET', 'POST'])
def live_capture():
    image_from_cam = trigger_webcam()
    image_path = '/home/vb/Documents/Aegis/Deep Learning/Project/static/image_01.png'
    imwrite(image_path, image_from_cam)
    img = image.img_to_array(image.load_img(image_path, target_size=(200,200)))
    test_images = [img]
    test_images = np.asarray(test_images)
    test_images = test_images * (1/255)
    val = model.predict(test_images)
    result = class_dict[val[0].argmax(axis=0)]

    return render_template("show.html", data = ["../static/" + 'image_01.png', result])



@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        global img
        global result
        img = image.img_to_array(image.load_img(f.filename, target_size=(200,200)))
        plt.imshow(image.load_img(f.filename, target_size=(200,200)))
        plt.axis("off")
        # plt.show()
        plt.savefig("/home/vb/Documents/Aegis/Deep Learning/Project/static/" + f.filename,)
        test_images = [img]
        test_images = np.asarray(test_images)
        test_images = test_images * (1/255)
        val = model.predict(test_images)
        result = class_dict[val[0].argmax(axis=0)]
        
#         print(f"Prediction : {result}") # class_dict[val]
#         return render_template("show.html", data = ["/home/vb/Documents/Aegis/Deep Learning/Project/static" + f.filename, result])
        return render_template("show.html", data = ["../static/" + f.filename, result])
if __name__ == '__main__':
    app.run()
