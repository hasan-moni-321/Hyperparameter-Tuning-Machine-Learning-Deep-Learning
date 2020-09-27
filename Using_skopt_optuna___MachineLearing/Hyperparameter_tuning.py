
##################################################################
# Hyperparameter tuning using different parameter
##################################################################


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection




if __name__ == "__main__":
    df = pd.read_csv("/home/hasan/Downloads/archive/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values
    
  
    classifer = RandomForestClassifier(n_jobs=-1)
    
   
    
    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 20, 1),
        "criterion":['gini', 'entropy']
    }
    
    model = model_selection.GridSearchCV(    #instead of RandomizedSearchCV  you can use GridSearchCV. GridSearchCV takes more time compared to RandomizedSearhCV 
        estimator = classifer,
        param_grid = param_grid,
        scoring = 'accuracy',
        verbose = 10,
        n_jobs = 1,
        cv=5
        )
    
    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())



