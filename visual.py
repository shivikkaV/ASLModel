from tkinter import *
import cv2 
from PIL import Image, ImageTk 
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from numpy import asarray
import time

#Importing the model
model = tf.keras.models.load_model("D:\Models\ASL.keras")
  
# Define a video capture object 
vid = cv2.VideoCapture(0) 
  
# Declare the width and height in variables 
width, height = 800, 1100
  
# Set the width and height 
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 
  
# Create a GUI app 
app = Tk() 
  
# Bind the app with Escape keyboard to 
# quit app whenever pressed 
app.bind('<Escape>', lambda e: app.quit()) 
  
# Create a label and display it on app 
label_widget = Label(app) 
label_widget.pack() 
  
# Create a function to open camera and 
# display it in the label_widget on app 

w = []
word = np.array(w)

CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z","space", "nothing"]

  
def open_camera(): 
    global frame
    _, frame = vid.read() 
    new_size = (800, 1100)
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # Convert image from one color space to other 
    captured_image = Image.fromarray(opencv_image)     # Capture the latest frame and transform to image 
    photo_image = ImageTk.PhotoImage(image=captured_image.resize(new_size))     # Convert captured image to photoimage 
    label_widget.photo_image = photo_image     # Displaying photoimage in the label 
    label_widget.configure(image=photo_image)     # Configure image in the label 
    label_widget.after(10, open_camera)     # Repeat the same process after every 10 seconds 
    prediction(captured_image) 
  
def prediction(captured_image):
    p = model.predict(prepare(captured_image))
    print(CATEGORIES[int(p[0][0])]) #
        
def prepare(captured_image):
    IMG_SIZE = 180
    image = asarray(captured_image)
    #img_array = cv2.imshow('', image)#error
    new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    np_array = np.asarray(new_array)
    np_final = np.expand_dims(np_array,axis=0)
    return np.resize(np_final, (1, 180, 180, 3))
    #return np_final.reshape(-3, IMG_SIZE, IMG_SIZE, 3) #error - the array keeps doubling the square of the image size, thus not allowing reshape property
        
  
# Create a button to open the camera in GUI app 
button1 = Button(app, text="Open Camera", command=open_camera) 
button1.pack() 
  
# Create an infinite loop for displaying app on screen 
app.mainloop() 