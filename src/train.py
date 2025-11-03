from pickle import dump

from sklearn.model_selection import train_test_split


def train_model(model, X, y, config):
    test_size = config["train"]["test_size"]
    random_state = config["train"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model.fit(X_train, y_train)

    model_path = config["output"]["model_path"]

    with open(model_path, "wb") as f:
        dump(model, f)

    print(model.feature_names_in_)

    print(f"Model saved to {model_path}")

    return model, X_test, y_test
