def predict_recession(model, scaler, latest_data):
    latest_scaled = scaler.transform(latest_data)
    prediction = model.predict(latest_scaled)
    return prediction[0][0] > 0.5