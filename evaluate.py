from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred, average="weighted"))
    print("\nReport:\n", classification_report(y_test, pred, zero_division=0))