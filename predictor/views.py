import os
import pandas as pd
import numpy as np
import joblib
from django.shortcuts import render
from predictor.feat_gen import generate_features_for_inference
from .encoders import le_driver, le_track, le_team  # if used

import xgboost as xgb

# Load model once (at server start)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = xgb.XGBRegressor()
model.load_model(os.path.join(BASE_DIR, "xgboost_model.json"))


def explain_prediction(features_df):
    explanation = []
    confi=[]

    grid = features_df["Starting Grid"].values[0]
    driver_avg = features_df["Driver_Avg_Pos"].values[0]
    track_avg = features_df["Driver_Track_History"].values[0]
    team_rel = features_df["TeamReliability"].values[0]
    driver_conf = features_df["DriverConfidence"].values[0]
    difficulty = features_df["Overtake Difficulty"].values[0]
    is_top_team = features_df["IsTopTeam"].values[0]

    low_confidence = False  

    # Conditions for explanation
    if grid <= 5:
        explanation.append("Good starting grid position.")
    elif grid >= 15 and difficulty > 0.2:
        explanation.append("Hard to overtake from the back.")
        low_confidence = True

    if driver_avg <= 6:
        explanation.append("Consistent high performer.")
    if track_avg <= 5:
        explanation.append("Historically strong at this track.")

    if team_rel > 0.8:
        explanation.append("Reliable constructor.")
    elif team_rel < 0.5:
        low_confidence = True

    if driver_conf > 0.9:
        explanation.append("Driver is in top form.")
    elif driver_conf < 0.7:
        low_confidence = True

    if is_top_team:
        explanation.append("Top-tier team support.")

    if not explanation:
        explanation.append("No standout factors.")

    # 💬 Add disclaimer if confidence is low
    if low_confidence:
        confi.append("⚠️ Prediction confidence is lower due to starting position or form.")
    else:
        confi.append("The Model Confidence is Normal")

    sentences = [sentence.strip() for item in explanation for sentence in item.split('.') if sentence.strip()]

    return sentences,confi



def predict_position(request):
    prediction = None
    explanation=[]
    confidence=[]
    csv_path = os.path.join(BASE_DIR, "feat_eng_f1.csv")
    df = pd.read_csv(csv_path)
    drivers = sorted(df["Driver"].unique())
    tracks = sorted(df["Track"].unique())

    if request.method == "POST":
        driver = request.POST.get("driver")
        track = request.POST.get("track")
        grid = int(request.POST.get("starting_grid"))

        features = generate_features_for_inference(driver, track, grid, df, year=2025)
        prediction = model.predict(features)[0]
        prediction=np.floor(prediction)
        explanation, confidence = explain_prediction(features)
    return render(request, "f1_form.html", {
        "drivers": drivers,
        "tracks": tracks,
        "prediction": prediction,
        "explanation": explanation,
        "confidence": confidence
    })


def about_us(request):
    return render(request,'about.html')

def home_page(request):
    return render(request,'f1_index.html')


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


class PredictAPIView(APIView):
    def post(self, request):
        data = request.data
        driver = data.get("driver")
        track = data.get("track")
        grid = data.get("starting_grid")

        if not (driver and track and grid):
            return Response({"error": "Missing data"}, status=status.HTTP_400_BAD_REQUEST)

        # Generate input features
        csv_path = os.path.join(BASE_DIR, "feat_eng_f1.csv")
        df = pd.read_csv(csv_path)
        features = generate_features_for_inference(driver, track, int(grid), df, year=2025)

        # Predict
        prediction = model.predict(features)[0]
        reason,confi = explain_prediction(features)


        return Response({
            "predicted_position": round(prediction, 2),
            "explanation": reason,
            "Confidence":confi
        })



