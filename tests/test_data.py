from src.data import download_data, prepare_data

def test_download_data():
    df = download_data()
    assert not df.empty
    assert "Recession" in df.columns

def test_prepare_data():
    df = download_data()
    (X_train, X_test, y_train, y_test), scaler = prepare_data(df)
    assert X_train.shape[1] == X_test.shape[1]