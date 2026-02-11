import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

# Lists of all features for each data type
num_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond','YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF','GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces','GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal','MoSold', 'YrSold']
nom_cols = ["MSZoning", "Street", "LotShape", "LandContour", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "Heating","Foundation", "SaleType", "SaleCondition"]
na_nom_cols = ["Alley", "MasVnrType", "GarageType", "MiscFeature"]
ord_cols = ["Utilities", "ExterQual", "ExterCond", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "PavedDrive"]
na_ord_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

# Lists for all categories for use with ordinal encoder sorted to match the ord_cols lists
ord_cols_cats = [["AllPub", "NoSewr", "NoSeWa", "ELO"], ["Ex", "Gd", "TA", "Fa","Po"], ["Ex", "Gd", "TA", "Fa","Po"],["Ex", "Gd", "TA", "Fa","Po"], ["Y", "N"], ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"], ["Ex", "Gd", "TA", "Fa","Po"], ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"], ["Y", "P", "N"]]
na_ord_cols_cats = [["Ex", "Gd","TA","Fa","Po","NA"], ["Ex", "Gd", "TA", "Fa", "Po", "NA"], ["Gd", "Av", "Mn", "No", "NA"], ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"], ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"], ["Ex", "Gd", "TA", "Masonry", "Fa", "Po", "NA"], ["Fin", "RFn", "Unf", "NA"],["Ex", "Gd","TA","Fa","Po","NA"],["Ex", "Gd","TA","Fa","Po","NA"], ["Ex", "Gd","TA","Fa","NA"], ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"], ["Elev", "Gar2", "Othr", "Shed", "TenC", "NA"]]
    

def ratio_pipeline():
    return make_pipeline(
            SimpleImputer(strategy="mean"),
            FunctionTransformer(column_ratio, feature_names_out=column_name)
    )

def column_ratio(X):
    num = X[:, [0]]
    den = X[:, [1]]
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den!=0)# Creates a zeros array of size num. Fills only the the positions in the array with the result when denominator != 0

def column_name(function_transformer, function_names_in):
    return ["ratio"]

def sum_ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        FunctionTransformer(column_sum_ratio, feature_names_out=column_sr_name)
    )

def column_sum_ratio(X):
    num = X[:, [0]] + X[:,[1]]
    den = X[:, [2]]
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den!=0)# Creates a zeros array of size num. Fills only the the positions in the array with the result when denominator != 0
    
def column_sr_name(function_transformer, function_names_in):
    return ["sum_ratio"]

preprocessing = ColumnTransformer([
    ("nums",make_pipeline(SimpleImputer(strategy="mean")), num_cols),
    ("nom_cols", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(sparse_output=False)), nom_cols),
    ("na_nom_cols", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(sparse_output=False)), na_nom_cols),
    ("ord_cols", make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder(categories=ord_cols_cats)), ord_cols),
    ("na_ord_cols", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OrdinalEncoder(categories=na_ord_cols_cats)),  na_ord_cols)],
    remainder="passthrough"
)

feature_eng = ColumnTransformer([
    ("TotalBsmtFinSF", make_pipeline(sum_ratio_pipeline()), ["nums__BsmtFinSF1", "nums__BsmtFinSF2", "nums__TotalBsmtSF"]),
    ("OverallQual_Cond", make_pipeline(ratio_pipeline()), ["nums__OverallQual","nums__OverallCond"]),
    ("BsmtBathPerSF", make_pipeline(sum_ratio_pipeline()),
    ["nums__BsmtFullBath", "nums__BsmtHalfBath", "nums__TotalBsmtSF"]),
    ("AbvGrBathPerSF", make_pipeline(sum_ratio_pipeline()), ["nums__FullBath", "nums__HalfBath", "nums__GrLivArea"]),
    ("AbvGrSFPerRoom", make_pipeline(sum_ratio_pipeline()), ["nums__1stFlrSF", "nums__2ndFlrSF", "nums__TotRmsAvgGrd"]),
    ("BmstQual_Cond", make_pipeline(ratio_pipeline()), ["na_ord_cols__BmstQual", "na_ord_cols__BsmtCond"]),
    ("GarageQual_Cond", make_pipeline(ratio_pipeline()), ["na_ord_cols__GarageQual", "na_ord_cols__GarageCond"]),
    ("GarageFinish_Qual_Cond", make_pipeline(sum_ratio_pipeline()), ["na_ord_cols__GarageQual", "na_ord_cols__GarageCond", "na_ord_cols__GarageFinish"])
    ],
    remainder="passthrough"
)

# Performs StandardScaler on all features
standard_scaling = ColumnTransformer([],
    remainder=StandardScaler()
)

# Performs MinMaxScaler on all features
minmax_scaler = ColumnTransformer([],
    remainder=MinMaxScaler(feature_range=(-1,1))
)