import cv2
import csv
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

label = input("Enter gesture label (e.g. fist, peace): ")


with open("gesture-data.csv", "a", newline="") as f:
    write = csv.writer(f)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            cv2.putText(image, "Press S to save", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                write.writerow(landmarks + [label])
                print(f"Saved {label}")

        cv2.imshow("Collect Data", image)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()