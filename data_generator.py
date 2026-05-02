import pandas as pd
import numpy as np

def generate_data(n=1000):
    data = pd.DataFrame({
        "traffic_density": np.random.randint(10, 200, n),
        "avg_speed": np.random.randint(20, 120, n),
        "visibility": np.random.randint(1, 10, n),
        "precipitation": np.random.randint(0, 2, n),
        "road_type": np.random.choice([0, 1], n),
        "harsh_braking": np.random.randint(0, 6, n),
    })

    # realistic rule-based risk
    risk = []
    for _, row in data.iterrows():
        score = (
            row["traffic_density"] * 0.3 +
            (120 - row["avg_speed"]) * 0.2 +
            (10 - row["visibility"]) * 2 +
            row["precipitation"] * 15 +
            row["harsh_braking"] * 3
        )
        if score < 50:
            risk.append(0)
        elif score < 100:
            risk.append(1)
        else:
            risk.append(2)

    data["accident_risk"] = risk
    return data

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/sample_data.csv", index=False)
    print("Dataset generated!")