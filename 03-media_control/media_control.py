from keras.models import load_model
from pynput.keyboard import Key, Controller
from collections import Counter
import time
import json
import cv2
import numpy as np

IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.9
COOLDOWN = 1


#load model
model = load_model("gesture_recognition_v2.keras")

label_names = []
with open("labels.json") as f:
    label_names = json.load(f)


print(label_names)
print(model.output_shape)


keyboard = Controller()
history = []


def predict_gesture(crop):

    img = cv2.resize(crop, (64, 64))
    img = img.astype(np.float32)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)

    idx = np.argmax(pred)

    confidence = pred[0][idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return "no_gesture", confidence
    
    return label_names[idx], confidence


#Media controls
def control_action(gesture):
    
    if gesture == "like":
        print("Volume Up")

        keyboard.press(Key.media_volume_up)
        keyboard.release(Key.media_volume_up)

    elif gesture == "dislike":
        print("Volume Down")

        keyboard.press(Key.media_volume_down)
        keyboard.press(Key.media_volume_down)

    elif gesture == "stop":
        print("Play/Pause")

        keyboard.press(Key.media_play_pause)
        keyboard.press(Key.media_play_pause)


#Camera
cap = cv2.VideoCapture(0)
last_action = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    
    box_size = 150
    
    x1 = w - box_size - 50
    y1 = h //  - box_size // 2

    x2 = x1 + box_size
    y2 = y1 + box_size

    crop = frame[y1:y2, x1:x2]

    #Prediction
    gesture, confidence = predict_gesture(crop)

    #Control gestures
    current_time = time.time()

    if gesture != "no_gesture" :

        if current_time - last_action > COOLDOWN:
            control_action(gesture)
            last_action = current_time

    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = f"{gesture}"

    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Media Controls", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
