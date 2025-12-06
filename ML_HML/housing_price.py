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
    return (
        OrdinalEncoder,
        Path,
        np,
        pd,
        plt,
        sns,
        tarfile,
        train_test_split,
        urllib,
    )


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
    return numeric_columns, numeric_columns_df


@app.cell
def _(housing_df):
    median = housing_df["total_bedrooms"].median()
    housing_df["total_bedrooms"] = housing_df["total_bedrooms"].fillna(median)
    housing_df.isnull().sum()
    return


@app.cell
def _(housing_df, numeric_columns, plt, sns):
    for _, column1 in enumerate(numeric_columns, 1):
        sns.boxplot(data=housing_df[column1], width=.5)
        plt.show()
    return


@app.cell
def _(housing_df, plt):
    housing_df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2, grid=True)
    plt.show()
    return


@app.cell
def _(housing_df, plt):
    housing_df.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                    s=housing_df["population"]/100, label="population",
                    c="median_house_value", cmap=plt.get_cmap("turbo"), colorbar=True,
                    legend=True, sharex=False, figsize=(12,8))
    plt.show()
    return


@app.cell
def _(numeric_columns_df, plt, sns):

    sns.pairplot(numeric_columns_df)

    # Display the plot
    plt.show()
    return


@app.cell
def _(numeric_columns_df, plt, sns):
    corr_matrix_numeric_columns = numeric_columns_df.corr()

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
def _(housing_df):
    housing_df.isnull().sum()
    return


@app.cell
def _(housing_df, np, plt, sns):
    # housing_comb_df = housing_df.copy()

    housing_df['rooms_per_house'] =  housing_df['total_rooms'] / housing_df['households']
    housing_df['bedrooms_per_room'] = housing_df['total_bedrooms'] / housing_df['total_rooms']
    housing_df['population_per_household'] = housing_df['population'] / housing_df['households']

    housing_num_df = housing_df.select_dtypes(include=[np.number])
    housing_num_df.head()
    corr_matrix_num = housing_num_df.corr()

    plt.figure(figsize=(8, 6))

    # 4. Plot the heatmap using Seaborn
    sns.heatmap(corr_matrix_num, 
                annot=True,      # Annotate cells with correlation values
                cmap='coolwarm', # Choose a color map (e.g., 'coolwarm', 'viridis')
                fmt=".2f",       # Format annotations to 2 decimal places
                linewidths=0.5   # Add lines to divide cells
               )

    plt.title("Correlation Heatmap")
    plt.show()
    return (housing_num_df,)


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
    return


@app.cell
def _(housing_df, train_test_split):
    # Train set size: 16512, Test set size: 4128
    train_set, test_set = train_test_split(housing_df, test_size=0.2, random_state=42)
    print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")
    return


@app.cell
def _(housing_df):
    housing_cat = housing_df[['ocean_proximity']]
    housing_cat.head(8)
    return (housing_cat,)


@app.cell
def _(OrdinalEncoder, housing_cat):
    ordinal_encoding = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoding.fit_transform(housing_cat)
    ordinal_encoding.categories_
    return


@app.cell
def _(housing_num_df):
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    housing_scaled = min_max_scaler.fit_transform(housing_num_df)

    return


if __name__ == "__main__":
    app.run()
