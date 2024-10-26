from sklearn.model_selection import train_test_split, GridSearchCV


class Modeling:
    def __init__(self):
        pass

    def split_data(self, df, target, test_size=0.25):
        X = df.drop(target, axis=1)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def grid_search(self, model, param_grid, cv, X_train, y_train):
        grid = GridSearchCV(model, param_grid, cv=cv)
        grid.fit(X_train, y_train)
        return grid.best_params_, grid.best_score_
