from sklearn.model_selection import  GridSearchCV, StratifiedKFold

from xgboost import XGBClassifier



def xgb_grid_mc(X_train, X_test, y_train, y_test, score='recall_macro', n_classes=3, kn=3):
    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        random_state=42,
        eval_metric='mlogloss',
        tree_method='hist'
    )
    param_grid = {
        'n_estimators': [100, 200],
        'min_child_weight': [1, 5],
        'gamma': [0, 0.5, 1.0],
        'learning_rate': [0.03, 0.1],
        'max_depth': [4, 6, 8],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    cv = StratifiedKFold(n_splits=kn, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=score,
        cv=cv,
        verbose=1,
        n_jobs=-1,
        error_score='raise'
    )
    grid.fit(X_train, y_train)
    print("Mejores parámetros:", grid.best_params_)
    print(f"Mejor {score}: {grid.best_score_:.4f}")
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)
    return y_pred, best, y_proba


