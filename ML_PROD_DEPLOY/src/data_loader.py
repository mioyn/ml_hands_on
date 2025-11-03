from sklearn.datasets import load_iris


def load_data():
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    print("Data loaded successfully.")
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y
