import cv2
import mediapipe as mp
import joblib
import numpy as np

model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                    
                )

            landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])

        prediction = model.predict([landmarks])[0]

        cv2.putText(image, prediction, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
    cv2.imshow("Gesture Recognition", image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()