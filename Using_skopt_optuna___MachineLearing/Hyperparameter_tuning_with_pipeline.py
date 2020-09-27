
######################################################
# Running Machine Learning problem using pipeline
######################################################



import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline





if __name__ == "__main__":
    df = pd.read_csv("/home/hasan/Downloads/archive/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values
    
    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = RandomForestClassifier(n_jobs=-1)
    
    classifier = pipeline.Pipeline([('scaling', scl), ('pca', pca), ('rf', rf)])
    
    
    param_grid = {
        "pca__n_components": np.arange(5,10),
        "rf__n_estimators": np.arange(100, 1500, 100),
        "rf__max_depth": np.arange(1, 20, 1),
        "rf__criterion":['gini', 'entropy']
    }
    
    model = model_selection.RandomizedSearchCV(    #instead of RandomizedSearchCV  you can use GridSearchCV. GridSearchCV takes more time compared to RandomizedSearhCV 
        estimator = classifier,
        param_distributions = param_grid,
        n_iter = 10,
        scoring = 'accuracy',
        verbose = 10,
        n_jobs = 1,
        cv=5
        )
    
    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())





# Note
###############################
#1. don't use pipeline because it decrease accuracy.

