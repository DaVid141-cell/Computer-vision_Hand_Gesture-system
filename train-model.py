import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

# Loads the data of the CSV file of the data collected
df = pd.read_csv("gesture-data.csv", header=None) 

print(df)                                       # display the data of the CSV file

X = df.iloc[:, :-1]                             # Landmarks of the hands
y = df.iloc[:, -1]                              # label of the gesture

print(X)                                        # prints the landmarks
print(y)                                        # prints the gesture name


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)                                 # Fits the data into the model x & y (21 hand landmarks & gesture label)

joblib.dump(model, "gesture_model.pkl")         # Saves the model into a Pickle File to use in the Recognizer
print("Model trained and saved")

