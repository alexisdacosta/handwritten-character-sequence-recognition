import gradio as gr
from tensorflow import keras
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np

model = keras.models.load_model('../notebooks/models/EMNIST.h5')

def classify(input):
    img = input.reshape(28, 28, 1)
    img = np.array(img)
    img = 255 - img
    cv2.imwrite('img.png', img)
    
    img = cv2.imread("img.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (28,28))
    img = 255 - img
    img_6 = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img_6 = image.img_to_array(img_6)
    img_6.reshape(28, 28)
    img_6 = np.expand_dims(img_6, axis=0)

    pred = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A' ,'B', 'C', 'D', 'E', 'F', 'G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t']

    pre = pred[np.argmax(model.predict(img_6))]
    # prediction = (model.predict(img_6) > 0.5).astype("int32")
    print(pre)
    
    prediction = model.predict(img_6).tolist()[0]
    
    for p in prediction:
        p = str(pred[int(p)])
        
    return {str(i): prediction[i] for i in range(10)}
    # return "This character is" + str(pre)

label = gr.outputs.Label(num_top_classes=3)
interface = gr.Interface(fn=classify, inputs="sketchpad", outputs=label, 
live=False)
interface.launch()