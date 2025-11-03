from sklearn.ensemble import RandomForestClassifier


def build_model(config):
    params = config["model"]["parameters"]
    model = RandomForestClassifier(**params)
    return model
