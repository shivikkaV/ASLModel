from tkinter import *
import cv2 
from PIL import Image, ImageTk 
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from numpy import asarray

#Try to import the model
try: 
    model = tf.keras.models.load_model("D:\Models\ASL.keras")
    print("Model Loaded Successfully")

#Exception - handles any exception related to loading the model
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
# Define a video capture object 
vid = cv2.VideoCapture(0) 
  
# Declare the width and height in variables 
width, height = 800, 600
  
# Set the width and height 
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 
  
# Create a GUI app 
app = Tk() 
app.title("ASL Recognition")

# Bind the app with Escape keyboard to 
# quit app whenever pressed 
app.bind('<Escape>', lambda e: app.quit()) 
  
# Create a label and display it on app 
label_widget = Label(app) 
label_widget.pack() 

# Create a prediction label to output the prediction
prediction_label = Label(app, text="Prediction")
prediction_label.pack()
  
CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z","space", "nothing"]

#Function to open camera
def open_camera(): 
    try: 
        global frame
        ret, frame = vid.read() 
        if not ret:
            print("Failed to grab frame")
            prediction_label.configure(text="Error: Could not read frame")
            app.after(1000, open_camera)
            return
        
        print("Frame read successfully")
        
        new_size = (800, 600)
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # Convert image from one color space to other 
        captured_image = Image.fromarray(opencv_image)     # Capture the latest frame and transform to image 
        photo_image = ImageTk.PhotoImage(image=captured_image.resize(new_size))     # Convert captured image to photoimage 
        label_widget.photo_image = photo_image     # Displaying photoimage in the label 
        label_widget.configure(image=photo_image)     # Configure image in the label 
        
        prediction(frame) # Calls prediction function

        label_widget.after(10, open_camera)     # Repeat the same process after every 10 seconds

    except Exception as e:
        print(f"Camera error: {e}")
        prediction_label.configure(text=f"Camera error: {e}")
        
#Prediction function - takes arg "frame"  
def prediction(frame):
    #Try making a prediction
    try:
        processed_image = prepare(frame)
        #Check if there is no processed image
        if processed_image is None:
            print("Error processing image")
            prediction_label.configure(text = "Error processing image")
            return
        
        p = model.predict(processed_image)
        predicted_class = np.argmax(p[0])
        confidence = str(p[0][predicted_class])
        print("Prediction: " + CATEGORIES[predicted_class] + ", Confidence: " + confidence) 
        prediction_label.configure(text=f"Prediction: {CATEGORIES[predicted_class]} ({confidence:.2f})")
    
    #Exception - Handles Prediction error
    except Exception as e:
        print(f"Prediction error: {e}")
        prediction_label.configure(text=f"Prediction error: {e}")

#Prepare function - Take arg "frame"
def prepare(frame): 
    try: 
        IMG_SIZE = 180
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        normalized_image = resized_image / 255.0
        return np.expand_dims(normalized_image, axis=0)
    
    #Exception - Handles Preparation errors
    except Exception as e:
        print(f"Image preparation error: {e}")
        return None
  
# Create a button to open the camera in GUI app 
button1 = Button(app, text="Open Camera", command=open_camera) 
button1.pack() 

status_label = Label(app, text="Ready - Press 'Open Camera' to start", fg="blue")
status_label.pack()
# Create an infinite loop for displaying app on screen 
app.mainloop() 

vid.release()
cv2.destroyAllWindows()
