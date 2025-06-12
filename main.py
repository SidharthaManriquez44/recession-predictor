from src.predictor import download_data, prepare_data, build_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def main():
    df = download_data()
    (X_train, X_test, y_train, y_test), scaler = prepare_data(df)

    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=0)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("Model Accuracy:", accuracy_score(y_test, y_pred))

    # Graphic
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.title('Evolution of loss')
    plt.show()

    latest_data = df.drop(columns=["Recession"]).iloc[-1:].values
    latest_scaled = scaler.transform(latest_data)
    prediction = model.predict(latest_scaled)

    print("There will be a recession soon?", "Yes" if prediction[0][0] > 0.5 else "No")


if __name__ == "__main__":
    main()
