
"""
Hyperparameter search for multiclass XGBoost with cross-validation.
Returns the best fitted model plus test predictions and probabilities.
"""

from sklearn.model_selection import  GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import mlflow.sklearn


def xgb_grid_mc(X_train, X_test, y_train, y_test, score='recall_macro', n_classes=3, kn=3,class_names=None):
    """
        Run a grid search over XGBoost hyperparameters for a multiclass problem.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training features.
        X_test : array-like of shape (m_samples, n_features)
            Test features to evaluate the best model on.
        y_train : array-like of shape (n_samples,)
            Training labels.
        y_test : array-like of shape (m_samples,)
            Test labels (only used to return predictions).
        score : str, default="recall_macro"
            Scoring metric passed to GridSearchCV (e.g., "accuracy", "f1_macro",
            "recall_macro", "precision_macro").
        n_classes : int, default=3
            Number of classes for the multiclass objective.
        kn : int, default=3
            Number of CV folds in StratifiedKFold.

        Returns
        -------
        y_pred : ndarray of shape (m_samples,)
            Predicted class labels on X_test produced by the best model.
        best : XGBClassifier
            Best estimator found by the grid search (already fitted on training data).
        y_proba : ndarray of shape (m_samples, n_classes)
            Class probabilities on X_test produced by the best model.
        """





    # Base estimator configured for multiclass probability outputs.
    xgb = XGBClassifier(
        objective='multi:softprob', # multiclass: returns class probabilities
        random_state=42,
        eval_metric='mlogloss',     # evaluation metric during training
        tree_method='hist'          # fast histogram algorithm
    )

    # Grid of hyperparameters to explore.
    param_grid = {
        'n_estimators': [100, 200],  # number of boosting rounds (trees)
        'min_child_weight': [1, 5],  # minimum sum of instance weight (Hessian) in a child
        'gamma': [0, 0.5, 1.0],  # minimum loss reduction to make a split
        'learning_rate': [0.03, 0.1],  # shrinkage rate
        'max_depth': [4, 6, 8],  # maximum depth of a tree
        'subsample': [0.8, 1.0],  # row sampling per tree
        'colsample_bytree': [0.8, 1.0],  # column sampling per tree
    }

    # Stratified CV preserves class proportions across folds.
    cv = StratifiedKFold(n_splits=kn, shuffle=True, random_state=42)

    # Grid search across the parameter space
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=score,
        cv=cv,
        verbose=1,
        n_jobs=-1,
        error_score='raise'
    )


    # Fit grid search on the training data.
    grid.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        sk_model=grid.best_estimator_,
        artifact_path="best_model"

    )


    print("Best parameters:", grid.best_params_)
    print(f"Best  {score}: {grid.best_score_:.4f}")



    # Retrieve the best trained model and evaluate on the test split.
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)



    return y_pred, best, y_proba


