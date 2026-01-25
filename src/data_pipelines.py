import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer

class Pipelines:
    def __init__(self):
        self.num_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond','YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF','GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces','GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal','MoSold', 'YrSold']
        self.nom_cols = ["MSZoning", "Street", "LotShape", "LandContour", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "Heating","Foundation", "SaleType", "SaleCondition"]
        self.na_nom_cols = ["Alley", "MasVnrType", "GarageType", "MiscFeature"]
        self.ord_cols = ["Utilities", "ExterQual", "ExterCond", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "PavedDrive"]
        self.na_ord_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]


    # Create ColumnTransformer that uses MinMaxScaler within each pipeline
    def create_column_transformer_mm(self):
        return make_column_transformer(
            (make_pipeline(SimpleImputer(strategy="mean"), MinMaxScaler(feature_range=(-1,1))), self.num_cols), # Numerical Features
            (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(sparse_output=False), MinMaxScaler(feature_range=(-1,1))), self.nom_cols), # Nominal Categories (NA not possible)
            (make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(sparse_output=False), MinMaxScaler(feature_range=(-1,1))), self.na_nom_cols), # Nominal Categories possible)
            (make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder(), MinMaxScaler(feature_range=(-1,1))), self.ord_cols), # Ordinal Categories (NA not possible)
            (make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OrdinalEncoder(), MinMaxScaler(feature_range=(-1,1))), self.na_ord_cols)
        )
    
    # Create ColumnTransformer that uses StandardScaler within each pipeline
    def create_column_transformer_ss(self):
        return make_column_transformer(
            (make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), self.num_cols), # Numerical Features
            (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(), StandardScaler(with_mean=False)), self.nom_cols), # Nominal Categories (NA not possible)
            (make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(), StandardScaler(with_mean=False)), self.na_nom_cols), # Nominal Categories (NA Possible)
            (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(), StandardScaler(with_mean=False)), self.ord_cols), # Orddinal Categories (NA not possible)
            (make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(), StandardScaler(with_mean=False)), self.na_ord_cols), # Ordinal Categoreis (NA possible)
        )
    
    def column_ratio(X):
        return X[:, [0]] / X[:, [1]]
    
    