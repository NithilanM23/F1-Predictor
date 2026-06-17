import pandas as pd
import numpy as np
import joblib
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the CSV file from this directory
csv_path = os.path.join(BASE_DIR, "feat_eng_f1.csv")

# Load it into a DataFrame
df_fin = pd.read_csv(csv_path)

from .encoders import le_driver,le_track

def generate_features_for_inference(driver_name, track_name, starting_grid, race_history_df=df_fin, year=2025):
    # Define confidence & reliability dictionaries (or import them globally)
    constructor_reliabilities = {
    "Ferrari": 0.94, "Red Bull": 0.93, "Mercedes": 0.94,
    "Aston Martin": 0.2, "Williams": 0.75, "Sauber": 0.3952755905511811,
    "RB": 0.4553903345724907, "McLaren": 0.93, "Alpine": 0.6018518518518519,
    "Haas": 0.34302325581395354,'Cadillac':0.3, 'Audi':0.43
        }

    driverconfidences = {
    "Lewis Hamilton": 0.9307114624505929, "George Russell": 0.9583333333333334, "Max Verstappen": 0.9842857142857143,
    "Sergio Pérez": 0.9333333333333333, "Charles Leclerc": 0.91, "Carlos Sainz": 0.9038461538461539,
    "Lando Norris": 0.9166666666666666, "Oscar Piastri": 0.94, "Esteban Ocon": 0.81,
    "Pierre Gasly": 0.83, "Yuki Tsunoda": 0.8, "Kimi Antonelli": 0.8, "Fernando Alonso": 0.88,
    "Lance Stroll": 0.8, "Valtteri Bottas": 0.90, "Franco Colapinto": 0.82,
    "Alexander Albon": 0.89, "Oliver Bearman": 0.83, "Kevin Magnussen": 0.87, "Nico Hülkenberg": 0.89
      }

    track_overtake_difficulty = {
    "Australia": 0.6,            # Street-like, medium difficulty
    "Bahrain": 0.3,              # Wide straights, DRS-friendly
    "China": 0.4,                # Long back straight, decent overtaking
    "Azerbaijan": 0.2,           # Easy due to long straight (Baku)
    "Spain": 0.7,                # Very difficult (technical, limited DRS)
    "Monaco": 0.9,               # Hardest overtaking track
    "Canada": 0.4,               # Multiple overtaking zones
    "France": 0.4,               # Paul Ricard is flat but allows overtakes
    "Austria": 0.3,              # Short track, lots of overtaking
    "Great Britain": 0.3,        # Silverstone – fast, flowing, decent overtaking
    "Germany": 0.5,              # Depends on circuit (Hockenheim usually)
    "Hungary": 0.8,              # Narrow and twisty – tough to pass
    "Belgium": 0.2,              # Spa – one of the best for overtaking
    "Italy": 0.3,                # Monza – high speed, easy DRS overtakes
    "Singapore": 0.85,           # Another very difficult street circuit
    "Russia": 0.5,               # Sochi – long straights, but dull racing
    "Japan": 0.7,                # Suzuka – tight corners, fast but limited overtaking
    "Mexico": 0.4,               # Long straight, then twisty sections
    "United States": 0.4,        # COTA – several overtaking spots
    "Brazil": 0.3,               # Interlagos – excellent for overtaking
    "Abu Dhabi": 0.7,            # Historically hard to pass (even with DRS)
    "Styria": 0.3,               # Same as Austria (Red Bull Ring)
    "70th Anniversary": 0.3,     # Silverstone layout
    "Tuscany": 0.6,              # Mugello – fast but hard to pass
    "Eifel": 0.5,                # Nürburgring GP – moderate
    "Portugal": 0.5,             # Algarve – flowing, but limited overtakes
    "Emilia Romagna": 0.75,      # Imola – historic but narrow
    "Turkey": 0.4,               # Turn 8 aside, overtaking is decent
    "Sakhir": 0.25,              # Outer layout, super short and overtaking-heavy
    "Netherlands": 0.8,          # Zandvoort – twisty and narrow
    "Qatar": 0.5,                # Lusail – mix of high-speed & tight turns
    "Saudi Arabia": 0.6,         # High-speed street circuit, tricky overtakes
    "Miami": 0.6,                # Narrow & stop-start
    "Las Vegas": 0.4,            # Long straights, slow corners = decent
    "Emilia-Romagna": 0.75,      # Same as Imola (duplication handled)
    'Barcelona-Catalunya': 0.7
      }



    overtake_difficulty = track_overtake_difficulty.get(track_name, 0.5)  # 0.5 = neutral


    # Filter past races for the given driver
    driver_races = race_history_df[race_history_df["Driver"] == driver_name]

    # Driver average position
    if not driver_races.empty:
        driver_avg_pos = driver_races["Position"].mean()
        team_name = driver_races["Team"].iloc[-1]  # latest team name for driver
    else:
        driver_avg_pos = race_history_df["Position"].mean()
        team_name = "Unknown"

    # Driver's average position at this track
    track_races = driver_races[driver_races["Track"] == track_name]
    if not track_races.empty:
        driver_track_history = track_races["Position"].mean()
    else:
        driver_track_history = driver_avg_pos

    # Encode driver and track
    try:
        driver_encoded = le_driver.transform([driver_name])[0]
    except:
        driver_encoded = -1  # unknown driver

    try:
        track_encoded = le_track.transform([track_name])[0]
    except:
        track_encoded = -1  # unknown track

    # TeamReliability
    team_reliability = constructor_reliabilities.get(team_name, 0.6)

    # DriverConfidence
    driver_confidence = driverconfidences.get(driver_name, 0.8)
    adjusted_grid_diff = (starting_grid - driver_avg_pos) * (1 + overtake_difficulty)
    # Lower means fewer places can realistically be gained
    position_gain_capacity = (20 - starting_grid) * (1 - overtake_difficulty)

    # IsTopTeam (static based on 2024 known strong performers)
    d_names = [
        'Max Verstappen', 'Isack Hadjar', 'Lando Norris', 'Kimi Antonelli',
        'Charles Leclerc', 'George Russell', 'Lewis Hamilton', 'Oscar Piastri'
    ]
    is_top_team = 1 if driver_name in d_names else 0

    # Final feature vector
    input_features = pd.DataFrame([{
        "DriverEncoded": driver_encoded,
        "TrackEncoded": track_encoded,
        "Starting Grid": starting_grid,
        "Driver_Avg_Pos": driver_avg_pos,
        "Driver_Track_History": driver_track_history,
        "IsTopTeam": is_top_team,
        "Year": year,
        "TeamReliability": team_reliability,
        "DriverConfidence": driver_confidence,
        "Overtake Difficulty": overtake_difficulty,
        "adjusted_grid_diff": adjusted_grid_diff,
        "position_gain_capacity": position_gain_capacity
    }])

    return input_features
