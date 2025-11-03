import yaml

from src import data_loader, evalyate, model, train


def main(config_path):
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load data
    X, y = data_loader.load_data()

    # Build model
    ml_model = model.build_model(config)

    # Train model
    trained_model, X_test, y_test = train.train_model(ml_model, X, y, config)

    # Evaluate model
    evalyate.evaluate_model(trained_model, X_test, y_test)


if __name__ == "__main__":
    main("configs/config.yaml")
