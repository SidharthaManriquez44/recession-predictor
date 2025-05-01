from src.predictor import descargar_datos, preparar_datos, construir_modelo
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def main():
    df = descargar_datos()
    (X_train, X_test, y_train, y_test), scaler = preparar_datos(df)

    model = construir_modelo(X_train.shape[1])
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=0)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("Precisión del modelo:", accuracy_score(y_test, y_pred))

    # Graphic
    plt.plot(history.history['loss'], label='Pérdida entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida validación')
    plt.legend()
    plt.title('Evolución de la pérdida')
    plt.show()

    latest_data = df.drop(columns=["Recession"]).iloc[-1:].values
    latest_scaled = scaler.transform(latest_data)
    prediction = model.predict(latest_scaled)

    print("¿Habrá recesión próximamente?", "Sí" if prediction[0][0] > 0.5 else "No")


if __name__ == "__main__":
    main()
