# Hand Gesture Recognition using MEdiaPipe and Machine Learning

## 1 . Project Introduction
This project is a real-time **hand gesture recognition system** built using **Computer Vision** and **Machine Learning**. It uses a webcam to detect hand landmarks, collects gesture data, trains a classification model, and then recognizes hand gesture live.The project is designed for learning purposes and demonstrates how hand gestures can be used for human-computer interaction.

---

## 2. Required Software and Libraries

To run this project, you need the following software and libraries installed:

**Software**
- Python 3.8 or higher
- Webcam (built-in or external)

**Python Libraries**
- See `requirements.txt` for exact versions:
    - mediapipe
    - opencv-python
    - scikit-learn
    - joblib
    - numpy
    - pandas
    - matplotlib
    - (and others included in `requirements.txt`)

---

## 3. Installation Steps

**Step 1: Clone the Repository**

Make sure Python is installed on your system. You can check by running:
```
python --version
```
after that you can clone the repo by doing:

```
git clone <repo-url>
cd <project-folder>
```

**Step 2: Create a Virtual Environment**
Run this following commands in your terminal:
```
python venv venv
```

**Step 3: Create a Virtual Environment**
```
venv\Scripts\activate
```

**Step 4: Install Required Libraries**
```
pip install -r requirements.txt
```
This will install all the required dependencies with compatible versions.

---

## 4. Project Files Explanation

`collect-data.py`
- Opens the webcam and detects hand landmarks using MediaPipe.
- Allows you to save gesture data into a CSV file by pressing **S**.
- Press **C** to create or change a gesture label.
- Used to build your dataset for gesture recognition.

`train-data.py`
- Reads the gesture data from `gesture-data.csv`.
- Trains a machine learning model (KNN classifier).
- Saves the trained model as `gesture_model.pkl`.

`gesture-recognizer.py`
- Load the trained gesture model.
- Uses the webcam to recognize hand gesture in real time.
- Displays the predicted gesture with confidence filtering and smoothing.

`gesture-data.csv`
- Stores the collected hand landmark data.
- Each row contains normalized landmark values and the corresponding gesture label.
- Used as traning data for the model.

`gesture_model.pkl`
- The trained machine learning model file.
- Generated after running `train-data.py`.
- Used by `gesture-recognizer.py` to predict gestures.

---

## 5. How to Run the Program

**Step 1: Collect Gesture Data**

Run this if you do not have a dataset or want to add new gestures:
```
python collect-data.py
```
- Press **S** to save gesture samples.
- Press **C** to create a new gesture label
- Press **ESC** to exit
- This will create the `gesture-data.csv` file for the gesture.

**Step 2: Train the Model**

After collecting data, train the model:
```
python train-data.py
```
- This will create `gesture_model.pkl`

**Step 3: Run the Gesture Recognition**

Finnaly, run the gesture recognizer:
```
python gesture-recognizer.py
```
- The webcam will open
- The program will predict and display the gestures that you have trained.