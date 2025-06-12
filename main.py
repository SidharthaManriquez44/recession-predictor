from src.data import download_data, prepare_data
from src.model import build_model
from src.predictor import predict_recession
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def main():
    df = download_data()
    (X_train, X_test, y_train, y_test), scaler = prepare_data(df)

    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=0)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("Model Accuracy:", accuracy_score(y_test, y_pred))

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()

    latest_data = df.drop(columns=["Recession"]).iloc[-1:].values
    will_recession = predict_recession(model, scaler, latest_data)
    print("Will there be a recession soon?", "Yes" if will_recession else "No")

if __name__ == "__main__":
    main()
