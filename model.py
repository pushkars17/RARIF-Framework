from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def train_model(data):
    X = data.drop("accident_risk", axis=1)
    y = data["accident_risk"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss'
)
    model.fit(X_train, y_train)

    return model, X_test, y_test