# 🚀 Recession Predictor
[![Python Tests](https://github.com/SidharthaManriquez44/csv_to_mysql/actions/workflows/python-app.yml/badge.svg)](https://github.com/SidharthaManriquez44/csv_to_mysql/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/github/SidharthaManriquez44/recession-predictor/graph/badge.svg?token=C8TA3LLQ7L)](https://codecov.io/github/SidharthaManriquez44/recession-predictor)


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

## 🧱 Project Structure

```plaintext
recession-predictor/
├── src/               # Core logic and configuration
├── test/              # Test of the logic
├── .gitignore                   
├── LICENSE            # License of the project
├── main.py            # Entry point
├── README.md
└── requirements.txt   # Requiremente of teh project
```

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
```bash
  pytest
```

## 🧹 Code Quality

   Lint, sort imports, and check types:
```bash
make lint
make format
make type-check
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## ✍️ Author
Developed by `Sidhartha Manriquez`.
Feel free to reach out or contribute to improve this project!
