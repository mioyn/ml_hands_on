import marimo

__generated_with = "0.18.2"
app = marimo.App()


@app.cell
def _():
    import tarfile
    import urllib.request
    from pathlib import Path

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder
    return Path, np, pd, plt, sns, tarfile, train_test_split, urllib


@app.cell
def _(Path, pd, tarfile, urllib):
    def load_housingdata():
        tarball_path = Path("datasets/housing.tgz")
        if not tarball_path.exists():
            Path("datasets").mkdir(parents=True, exist_ok=True)
            url = "https://github.com/ageron/data/raw/main/housing.tgz"
            urllib.request.urlretrieve(url, tarball_path)
            print("Downloaded housing dataset.")
            with tarfile.open(tarball_path) as housing_tarball:
                housing_tarball.extractall(path="datasets")
                print("Extracted housing dataset.")
        housing_csv_path = Path("datasets/housing/housing.csv")
        return pd.read_csv(housing_csv_path)

    housing_df = load_housingdata()
    housing_df.head()
    return (housing_df,)


@app.cell
def _(housing_df):

    housing_df.info()
    housing_df.describe()
    return


@app.cell
def _(housing_df):
    housing_df['ocean_proximity'].value_counts()
    return


@app.cell
def _(housing_df, np, plt, sns):
    # housing_df.hist(bins=50, figsize=(20,15))
    # plt.show()
    numeric_columns_df = housing_df.select_dtypes(include=[np.number])
    numeric_columns = numeric_columns_df.columns

    for i, column in enumerate(numeric_columns, 1):
        sns.histplot(data=housing_df[column], bins=50, kde=True)
        plt.show()
    return


@app.cell
def _(housing_df, np, pd, plt):
    housing_df['income_cat'] = pd.cut(
        housing_df['median_income'],
        bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    plt.subplots( 1, 1, figsize=(12, 6))
    housing_df['income_cat'].hist()
    plt.show()
    return


@app.cell
def _(housing_df, train_test_split):
    strat_train_set, strat_test_set = train_test_split(
        housing_df,
        test_size=0.2,
        stratify=housing_df['income_cat'],
        random_state=42
    )

    strat_train_set["income_cat"].value_counts() / len(strat_train_set)
    return strat_test_set, strat_train_set


@app.cell
def _(strat_test_set, strat_train_set):
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return


@app.cell
def _(plt, strat_train_set):
    housing_c = strat_train_set.copy()
    housing_c.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2, grid=True)
    plt.show()
    return (housing_c,)


@app.cell
def _(housing_c, plt):
    housing_c.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                    s=housing_c["population"]/100, label="population",
                    c="median_house_value", cmap=plt.get_cmap("turbo"), colorbar=True,
                    legend=True, sharex=False, figsize=(12,8))
    plt.show()
    return


@app.cell
def _(housing_c, plt, sns):
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

    sns.pairplot(housing_c[attributes])

    # Display the plot
    plt.show()
    return (attributes,)


@app.cell
def _(attributes, housing_c, plt, sns):
    corr_matrix_numeric_columns = housing_c[attributes].corr()

    plt.figure(figsize=(8, 6))

    # 4. Plot the heatmap using Seaborn
    sns.heatmap(corr_matrix_numeric_columns, 
                annot=True,      # Annotate cells with correlation values
                cmap='coolwarm', # Choose a color map (e.g., 'coolwarm', 'viridis')
                fmt=".2f",       # Format annotations to 2 decimal places
                linewidths=0.5   # Add lines to divide cells
               )

    plt.title("Correlation Heatmap")
    plt.show()
    return


@app.cell
def _(strat_train_set):
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    return housing, housing_labels


@app.cell
def _(np):

    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.preprocessing import OneHotEncoder

    from sklearn.cluster import KMeans

    class ClusterSimilarity(BaseEstimator, TransformerMixin):
        def __init__(self, n_clusters=10, random_state=None, gamma=1.0):
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.gamma = gamma
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

        def fit(self, X, y=None, sample_weight=None):
            self.kmeans.fit(X, sample_weight=sample_weight)
            return self

        def transform(self, X):
            return rbf_kernel(X, self.kmeans.cluster_centers_, gamma=self.gamma)

        def get_feature_names_out(self, input_features=None):
            return [f"cluster {i} similarity" for i in range(self.n_clusters)]

    def column_ratio(x):
        return x[:, [0]] / x[:, [1]]

    def ratio_name(function_transformer, feature_names_in):
        return ["ratio"]

    def ratio_pipeline():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(column_ratio, feature_names_out=ratio_name),
            StandardScaler())

    logging_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler())

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    cluster_simil = ClusterSimilarity(n_clusters=10, random_state=42, gamma=1.0)

    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler())

    preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_household", ratio_pipeline(), ["population", "households"]),
        ("log", logging_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("geo", cluster_simil, ["longitude", "latitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object))
    ], remainder=default_num_pipeline)
    return Pipeline, make_pipeline, preprocessing


@app.cell
def _(housing, preprocessing):
    housing_prepared = preprocessing.fit_transform(housing)
    housing_prepared.shape
    return


@app.cell
def _(preprocessing):
    preprocessing.get_feature_names_out()
    return


@app.cell
def _(housing, housing_labels, make_pipeline, np, preprocessing):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    lin_reg = make_pipeline(
        preprocessing,
        LinearRegression()
    )
    lin_reg.fit(housing, housing_labels)
    housing_predictions = lin_reg.predict(housing)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    print("RMSE:", np.sqrt(lin_mse))

    mean_avg_error = np.mean(np.abs(housing_predictions - housing_labels))
    print("MAE:", mean_avg_error)
    return (mean_squared_error,)


@app.cell
def _(
    housing,
    housing_labels,
    make_pipeline,
    mean_squared_error,
    np,
    preprocessing,
):
    from sklearn.tree import DecisionTreeRegressor

    tree_reg = make_pipeline(
        preprocessing,
        DecisionTreeRegressor(random_state=42)
    )
    tree_reg.fit(housing, housing_labels)
    housing_predictions_tree = tree_reg.predict(housing)
    tree_mse = mean_squared_error(housing_labels, housing_predictions_tree)
    print("RMSE:", np.sqrt(tree_mse))
    mean_avg_error_tree = np.mean(np.abs(housing_predictions_tree - housing_labels))
    print("MAE:", mean_avg_error_tree)
    return (tree_reg,)


@app.cell
def _(housing, housing_labels, tree_reg):
    from sklearn.model_selection import cross_val_score
    scores = -cross_val_score(
        tree_reg,
        housing,
        housing_labels,
        scoring="neg_root_mean_squared_error",
        cv=10
    )

    print("Scores:", scores)
    print("Mean:", scores.mean())
    return (cross_val_score,)


@app.cell
def _(cross_val_score, housing, housing_labels, make_pipeline, preprocessing):
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = make_pipeline(
        preprocessing,
        RandomForestRegressor(random_state=42)
    )
    forest_rmse_scores = -cross_val_score(
        forest_reg,
        housing,
        housing_labels,
        scoring="neg_root_mean_squared_error",
        cv=10
    )
    print("Scores:", forest_rmse_scores)
    print("Mean:", forest_rmse_scores.mean())
    return (RandomForestRegressor,)


@app.cell
def _(Pipeline, RandomForestRegressor, housing, housing_labels, preprocessing):
    from sklearn.model_selection import GridSearchCV

    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42))
    ])
    param_grid = [
        {'preprocessing__geo__n_clusters': [5, 8, 10],
        'random_forest__max_features': [4, 6, 8]},
        {'preprocessing__geo__n_clusters': [10, 15],
        'random_forest__max_features': [6, 8, 10]},
    ]

    grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid_search.fit(housing, housing_labels)
    return full_pipeline, grid_search


@app.cell
def _(grid_search):
    grid_search.best_params_
    return


@app.cell
def _(grid_search, pd):
    cv_res = pd.DataFrame(grid_search.cv_results_)
    cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
    cv_res
    return


@app.cell
def _(full_pipeline, housing, housing_labels):
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint

    param_distribs = {
        'preprocessing__geo__n_clusters': randint(low=3, high=50),
        'random_forest__max_features': randint(low=2, high=20)}

    rnd_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_distribs,
        n_iter=10,
        cv=3,
        scoring='neg_root_mean_squared_error',
        random_state=42
    )
    rnd_search.fit(housing, housing_labels)
    rnd_search.best_params_
    return (rnd_search,)


@app.cell
def _(rnd_search):
    final_model = rnd_search.best_estimator_
    feature_importances = final_model.named_steps['random_forest'].feature_importances_
    feature_names = final_model.named_steps['preprocessing'].get_feature_names_out()
    sorted(zip(feature_importances, feature_names), reverse=True)
    return (final_model,)


@app.cell
def _(final_model, strat_test_set):
    from sklearn.metrics import root_mean_squared_error

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    final_predictions = final_model.predict(X_test)
    final_rmse = root_mean_squared_error(y_test, final_predictions)
    print("Final RMSE on Test Set:", final_rmse)
    return


if __name__ == "__main__":
    app.run()
