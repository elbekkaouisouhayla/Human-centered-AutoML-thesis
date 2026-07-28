import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=2026)

# Average CV score on the training set was: 0.8119583177306078
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=2, min_child_weight=15, n_estimators=100, n_jobs=1, subsample=0.9500000000000001, verbosity=0)),
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        ),
        SelectFromModel(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.3, n_estimators=100), threshold=0.05)
    ),
    RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.8, min_samples_leaf=6, min_samples_split=3, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 2026)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
