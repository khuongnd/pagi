from catboost import CatBoostClassifier


def get_CatBoostClassifier(iterations=1000):
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.01,
        min_data_in_leaf=10,
        eval_metric='AUC'
    )


if __name__ == "__main__":
    from catboost.datasets import titanic
    import numpy as np
    train_df, test_df = titanic()

    train_df.head()
    train_df.fillna(-999, inplace=True)
    test_df.fillna(-999, inplace=True)
    X = train_df.drop('Survived', axis=1)
    y = train_df.Survived
    print(X.dtypes)

    categorical_features_indices = np.where(X.dtypes != np.float)[0]
    from sklearn.model_selection import train_test_split

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)

    X_test = test_df

    model = get_CatBoostClassifier()

    model.fit(
        X_train, y_train,
        cat_features=categorical_features_indices,
        eval_set=(X_validation, y_validation),
        plot=True
    )

    preds = model.predict_proba(X_validation)[:, 1]

    from pagi.utils.metrics import get_binary_classification_metrics

    print(y_validation.values.shape, preds.shape)
    print(get_binary_classification_metrics(y_validation.values.astype(np.float), preds))
