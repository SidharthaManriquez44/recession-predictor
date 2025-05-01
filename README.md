# 🚀 Recession Predictor

**Recession Predictor** is a machine learning project that aims to forecast potential future recessions using macroeconomic indicators and a neural network built with TensorFlow and Keras.

This project is developed in **Python** and uses libraries such as `pandas`, `scikit-learn`, `TensorFlow`, and `pandas_datareader` to gather and process data from the U.S. Federal Reserve (FRED).

---

## 📌 Features

- ✅ Automatic download of key economic indicators from FRED:
  - GDP
  - Unemployment Rate
  - Federal Funds Interest Rate
  - Inflation Index (CPI)
- ✅ Preprocessing pipeline:
  - Fill missing values
  - Feature engineering (e.g., GDP Growth, Inflation Rate)
  - Feature scaling with `StandardScaler`
- ✅ Neural network:
  - 3 Dense layers with ReLU activation
  - Dropout layers to reduce overfitting
  - Binary classification using sigmoid output
- ✅ Model training and evaluation
- ✅ Visualization of training loss and validation loss
- ✅ Final prediction on most recent data

---

## 🧠 Technologies Used

- Python 3.9+
- TensorFlow / Keras
- Scikit-learn
- Pandas
- Pandas DataReader
- Matplotlib

---

## 📁 Project Structure

recession_predictor/ │ ├── src/ # Core logic and modules │ └── predictor.py ├── main.py # Main entry point for training and prediction ├── requirements.txt # Dependencies ├── README.md └── .gitignore

## ▶️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/recession-predictor.git
   cd recession-predictor
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the main script:
    ```bash
   python main.py
   ```
   
## 🧪 Testing
(Coming soon...) Unit tests will be added in future versions using pytest.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## ✍️ Author
Developed by `Sidhartha Manriquez`.
Feel free to reach out or contribute to improve this project!
