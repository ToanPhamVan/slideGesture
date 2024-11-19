import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# from sklearn.metrics import accuracy_score  # Accuracy metrics
import pickle
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

df = pd.read_csv("coords.csv")
X = df.drop("class", axis=1)  # features
y = df["class"]  # target value
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234
)
pipelines = {
    "lr": make_pipeline(StandardScaler(), LogisticRegression()),
    "rc": make_pipeline(StandardScaler(), RidgeClassifier()),
    "rf": make_pipeline(StandardScaler(), RandomForestClassifier()),
    "gb": make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model
with open("body_language.pkl", "wb") as f:
    pickle.dump(fit_models["rf"], f)
warnings.filterwarnings("ignore")

# Alternatively, suppress warnings through logging
logging.getLogger("py.warnings").setLevel(logging.ERROR)

# Khởi tạo các mô hình Mediapipe
mp_holistic = mp.solutions.holistic

# cap = cv2.VideoCapture(0)

# # Khởi động pyttsx3
# # engine = pyttsx3.init()

# # Biến theo dõi class trước đó
previous_class = None