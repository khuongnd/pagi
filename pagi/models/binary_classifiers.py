from catboost import CatBoostClassifier


def get_CatBoostClassifier(iterations=1000, learning_rate=0.01, min_data_in_leaf=30, eval_metric='AUC', cat_features=None):
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.01,
        min_data_in_leaf=min_data_in_leaf,
        eval_metric=eval_metric,
        cat_features=cat_features
    )

