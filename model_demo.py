import joblib
import numpy as np

def predict_sample(sample):
    bundle = joblib.load("best_model.joblib")
    model = bundle["model"]
    scaler = bundle["scaler"]

    sample_scaled = scaler.transform([sample])
    prediction = model.predict(sample_scaled)[0]

    label_map = {0: "MALIGNANT", 1: "BENIGN"}
    return label_map[prediction]


if __name__ == "__main__":
    # example dummy input with 30 features
    test_sample = np.random.rand(30)
    print("Prediction Output:", predict_sample(test_sample))
