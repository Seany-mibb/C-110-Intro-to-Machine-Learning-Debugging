# import the opencv library
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()

    #1. Resizing the image
    img = cv2.resize(frame, (224, 224))

    #2.Converting the image into Numpy array and increace dimension
    #dtype is datatype and np.float32 converts img into float numbers (decimal numbers)
    test_image = np.array(img, dtype = np.float32)
    test_image = np.expand_dims(test_image, axis=0)

    #3.Normalizing the array values of the image
    normalized_image = test_image/255.0

    #Predict result
    prediction = model.predict(normalized_image)

    print(f"Prediction: {prediction}")

    cv2.imshow("Result", frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()