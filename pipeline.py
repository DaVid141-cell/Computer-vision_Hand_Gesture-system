import warnings
from collect_data import collect
from train_model import training
from gesture_recognizer import recognizer

warnings.filterwarnings("ignore", category=UserWarning) # this just to ignore the warning message of the mediapipe since i'm using an older version of it.


print("Step 1: Collecting gesture data...")
collect()

print("Step 2: Training model...")
training()

print("Step 3: Starting gesture recognition...")
recognizer()