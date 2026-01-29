import marimo

__generated_with = "0.18.2"
app = marimo.App()


@app.cell
def _():
    from sklearn.datasets import fetch_openml
    """Load the MNIST dataset from OpenML.

    Returns:
        X: Features (images) as a numpy array.
        y: Labels (digits) as a numpy array.
    """
    mnist = fetch_openml('mnist_784', as_frame=False)
    X, y = mnist.data, mnist.target
    X
    return X, y


@app.cell
def _(y):
    y
    return


@app.cell
def _(X, y):
    import matplotlib.pyplot as plt
    def plot_digit(data):
        """Plot a single digit from the MNIST dataset.

        Args:
            data: A 1D array of 784 pixel values representing a digit.
        """
        image = data.reshape(28, 28)
        plt.imshow(image, cmap='binary')
        plt.axis('off')

    plot_digit(X[0])
    plt.show()

    print("Label:", y[0])
    return


@app.cell
def _(X, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_test, y_train


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDClassifier
    import numpy as np

    # preprocessing = Pipeline(steps=[
    #     ('scaler', StandardScaler()),
    # ])

    def scale_pixels(X):
        # Scale pixel values to [0.0, 1.0] range
        return np.asarray(X, dtype=np.float64) / 255.0

    sgd_pipeline = Pipeline(steps=[
        ('normalizer', FunctionTransformer(scale_pixels)),    
        ("classifier", SGDClassifier(random_state=42))
    ])

    return Pipeline, SGDClassifier, sgd_pipeline


@app.cell
def _(X_train, sgd_pipeline, y_train):
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import loguniform

    param_distributions = {
        'classifier__alpha': loguniform(1e-5, 1e-2),
        'classifier__penalty': ['l2', 'l1', 'elasticnet'],
        'classifier__loss': ['hinge', 'log_loss'],
        'classifier__max_iter': [1000, 2000, 3000],
        'classifier__learning_rate': ['optimal', 'adaptive'],
        'classifier__eta0': loguniform(1e-4, 1e-1)
    }

    random_search = RandomizedSearchCV(
        estimator=sgd_pipeline,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    print("Running RandomizedSearchCV...")
    random_search.fit(X_train, y_train)
    print("Search complete.")
    return


@app.cell
def _(Pipeline, SGDClassifier, X_test_scaled, X_train, y_test, y_train):
    from sklearn.model_selection import GridSearchCV
    sgd_pipeline = Pipeline([
        ("sgd_classifier", SGDClassifier(random_state=42))
    ])

    # Define the parameter grid for GridSearchCV
    param_grid_sgd = {
        'loss': ['hinge', 'log_loss', 'modified_huber'], # Try different loss functions
        'penalty': ['l2', 'l1', 'elasticnet'],          # Try different regularization penalties
        'alpha': [0.0001, 0.001, 0.01, 0.1],            # Regularization strength
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=sgd_pipeline,
        param_grid=param_grid_sgd,
        cv=5,            # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1        # Use all available cores
    )

    # Fit the grid search to the scaled training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters and score
    print(f"Best parameters (GridSearchCV): {grid_search.best_params_}")
    print(f"Best CV accuracy (GridSearchCV): {grid_search.best_score_:.3f}")
    print(f"Test set accuracy (GridSearchCV): {grid_search.score(X_test_scaled, y_test):.3f}")

    return (sgd_pipeline,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
