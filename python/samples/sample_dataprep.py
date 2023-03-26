import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection as mds
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
# from sklearn.preprocessing import Imputer  # Deprecated
# from pandas.plotting import scatter_matrix

print('sklearn version: ' + sklearn.__version__)

path = r"F:\Sebastien\OneDrive\Work\ML\Geron - Hands-On Machine Learning\datasets\housing"
file = os.path.join(path, 'housing.csv')
housing = pd.read_csv(file)

# View information from the dataframe
# print(housing.head())
# print()
# print(housing.info())
# print()
# print(housing["ocean_proximity"].value_counts())
# print()
# print(housing["longitude"].describe())
print()
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()


# Generate training and testing sets. Equivalent to sklearn's train_test_split
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# Create a new category (strata) and use stratified sampling
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# Classic way of generating training and testing sets
# train_set, test_set = split_train_test(housing, 0.20)
train_set, test_set = mds.train_test_split(housing, test_size=0.20, random_state=42)

# Stratified sampling
splitter = mds.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print()
# print('Original set')
# print(housing["income_cat"].value_counts() / len(housing))
# print('Classic training set')
# print(train_set["income_cat"].value_counts() / len(train_set))
# print('Classic test set')
# print(test_set["income_cat"].value_counts() / len(test_set))
# print('SS training set')
# print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
# print('SS test set')
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# Remove strata column
for lset in (strat_train_set, strat_test_set):
    lset.drop(["income_cat"], axis=1, inplace=True)

# Calculate correlations
corr_matrix = housing.corr()
# print(corr_matrix)

# Sort correlations
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Plot scatter
# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

# Plot only one
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

# Prepare variables and labels
housing = strat_test_set.drop("median_house_value", axis=1)  # drop() creates a copy
housing_labels = strat_test_set["median_house_value"]

# User Imputer class to fill in missing values with median
imputer = SimpleImputer(strategy="median")
# imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)  # Remove non-numerical
imputer.fit(housing_num)  # "fit", i.e. calculate medians
# print(imputer.statistics_)
# print(housing_num.median().values)
X = imputer.transform(housing_num)  # "transform", i.e. fill the n/a with median
housing_tr = pd.DataFrame(X, columns=housing_num.columns)  # Put back into a dataframe

# Handling text and categorical attributes
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
# print(housing_cat[:5])
housing_cat_encoded = encoder.fit_transform(housing_cat)
# print(encoder.classes_)
# print(housing_cat_encoded[:5])

# Problem here is that the integer values have no meaning, but Mathematically they will be
# interpreted as closer if numerically closer, which is not necessarily true. So instead
# we put them in "one-hot encoding" of only 0s or 1s per class.
encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# print(housing_cat_1hot[0, 0])

# Create a custom transformer class
room_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, x, y=None):
        x, y
        return self

    def transform(self, x, y=None):
        y
        rooms_per_household = x[:, bedrooms_ix] / x[:, household_ix]
        population_per_household = x[:, population_ix] / x[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = x[:, bedrooms_ix] / x[:, room_ix]
            return np.c_[x, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[x, rooms_per_household, population_per_household]


# Use the class
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# Pipeline
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler()),
                         ])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# Dataframe selector
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, x, y=None):
        x, y
        return self

    def transform(self, x):
        return x[self.attribute_names].values


# Combine pipelines
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                         ('imputer', SimpleImputer(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler()),
                         ])

cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                         ('cat_encoder', OneHotEncoder(sparse=False))
                         ])
# ('label_binarizer', LabelBinarizer())

full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),
                                               ("cat_pipeline", cat_pipeline),
                                              ])
housing_prepared = full_pipeline.fit_transform(housing)
# housing_prepared = cat_pipeline.fit_transform(housing)

# ########################################################################################
# ################ Training, K-fold validation, and Prediction ###########################
# ########################################################################################
# Linear regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions:\t", lin_reg.predict(some_data_prepared))
# print("Labels:\t\t", list(some_labels))
housing_predictions = lin_reg.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predictions)
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print("LinReg RMSE:\t", np.sqrt(mse))
print("LinReg K-fold mean:\t", rmse_scores.mean())
print("LinReg K-fold stDev:\t", rmse_scores.std())

# Decisition tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predictions)
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print()
print("Tree RMSE:\t", np.sqrt(mse))
print("Tree K-fold mean:\t", rmse_scores.mean())
print("Tree K-fold stDev:\t", rmse_scores.std())

# Random forest
forest_reg = RandomForestRegressor(10)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predictions)
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print()
print("Random forest RMSE:\t", np.sqrt(mse))
print("Random forest K-fold mean:\t", rmse_scores.mean())
print("Random forest K-fold stDev:\t", rmse_scores.std())

# Save a model in sklearn
# model = forest_reg
# file_name = "forest_reg.pkl"
# joblib.dump(model, file_name)
# model_load = joblib.load(file_name)
# housing_predictions = model_load.predict(housing_prepared)
# mse = mean_squared_error(housing_labels, housing_predictions)
# scores = cross_val_score(model_load, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
# print()
# print("Loaded model RMSE:\t", np.sqrt(mse))
# print("Loaded model K-fold mean:\t", rmse_scores.mean())
# print("Loaded model K-fold stDev:\t", rmse_scores.std())

# Grid search
param_grid = [{'n_estimators': [50, 60, 80], 'max_features': [4, 6, 8]},
              {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 5]}]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
print()
print('Grid search')
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Evaluate on the test set
final_model = grid_search.best_estimator_
x_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)
mse = mean_squared_error(final_predictions, y_test)
print()
print("Final model RMSE:\t", np.sqrt(mse))


