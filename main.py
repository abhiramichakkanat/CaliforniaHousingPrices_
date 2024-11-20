import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_absolute_percentage_error,r2_score
from Modules.correlation_to_target import CorrelationToTarget
from Modules.loading import Loading
from Modules.find_null_values import FindNullValues
from Modules.fillna import FillNa
from Modules.one_hot_encoding import OneHotEncoding
from Modules.correlation_to_target import CorrelationToTarget
from Modules.outliers import Outlier
from Modules.vif import VIF
from Modules.train_test_split import TrainTestSplit
from Modules.standardization import Standardization
from Modules.model_fitting import ModelFitting
from Modules.evaluation import Evaluation
from Modules.correlation_to_each_other import CorrelationToEachOther
from Modules.stat_models_ols import StatModelOLS
import warnings
from Modules.drop_columns import DropColumns
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)


read_method_=pd.read_csv
path_=r"C:\Users\Admin\Desktop\Polestar_Work\Day-1_Preprocesing_2\Day-1_Preprocesing\housing\housing.csv"
loaderObject = Loading(path_,read_method_)
data_ = loaderObject.data

findNullValuesObject=FindNullValues(data_)
numericalNullColumns=findNullValuesObject.numericalNullColumns
categoricalNullColumns=findNullValuesObject.categoricalNullColumns


window=10
fillStrategy={'mean':[], 'drop':[],'median':[],'mode':[],'ffill':[],'bfill':[],'interpolate':[],'movingAverage':['total_bedrooms']}
fillNaObject = FillNa(data_, fillStrategy,window=window)
filledData_=fillNaObject.filledData


dtype=float
dropFirstColumn=True
categoricalColumns_=["ocean_proximity"]
oneHotEncodedDataObject=OneHotEncoding(filledData_,categoricalColumns_,dtype,dropFirstColumn)
oneHotEncodedData_=oneHotEncodedDataObject.oneHotEncodedData

method_outlier_="iqr"
zscoreThreshold=3
Q1=0.25
Q2=0.75
multiplierLB=1.5
multiplierUB=1.5
outlierObject=Outlier(oneHotEncodedData_,method_outlier_,zscoreThreshold=zscoreThreshold,quantile_lower=Q1,quantile_upper=Q2,multiplierLb=multiplierLB,multiplierUb=multiplierUB)
cleanedData_=outlierObject.cleanedData

cleanedData_['rooms_per_household']=cleanedData_['total_rooms']/cleanedData_['households']
cleanedData_['bedrooms_per_room']=cleanedData_['total_bedrooms']/cleanedData_['total_rooms']
cleanedData_['population_per_household']=cleanedData_['population']/cleanedData_['households']


target='median_house_value'
columnsForCorrelation=['longitude','latitude','median_house_value','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity_INLAND','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN','rooms_per_household','bedrooms_per_room','population_per_household']
correlationToTargetObject=CorrelationToTarget(columnsForCorrelation,cleanedData_,target)
correlationToTarget=correlationToTargetObject.correlationToTargetCorrMatrix
correlationToTargetPearson=correlationToTargetObject.pTestResults
correlationToTargetKandallTau=correlationToTargetObject.kendallTauResults


targetColumn_=['median_house_value']
X = cleanedData_.drop(columns=targetColumn_)  
y = cleanedData_[targetColumn_]


columnsToDrop=['total_rooms','total_bedrooms']
dropColumnsObject=DropColumns(X,columnsToDrop)
data=dropColumnsObject.droppedData

columnsForCorrelation=['longitude','latitude','median_house_value','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity_INLAND','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN','rooms_per_household','bedrooms_per_room','population_per_household']
correlationToEachOtherObject=CorrelationToEachOther(columnsForCorrelation,cleanedData_)
correlatedPairs=correlationToEachOtherObject.correlatedPairs
pTestIndependentPairs=correlationToEachOtherObject.pTestResultsofIndependentPairs
kendalTauIndependentPairs=correlationToEachOtherObject.kendalTauResultsOfIndependentPairs


testSize=0.2
randomState=15
trainTestSplitObject=TrainTestSplit(data, y ,testSize=testSize, random_State=randomState)


xtrain_=trainTestSplitObject.xTrain
ytrain_=trainTestSplitObject.yTrain
xtest_=trainTestSplitObject.xTest
ytest_=trainTestSplitObject.yTest


method_ = 'yeo-johnson'
standardizationObject = Standardization(xtrain_, xtest_, method_)
standardizedXTrain=standardizationObject.xTrain_transformed
standardizedXTest=standardizationObject.xTest_transformed

columnsForCorrelation=['longitude','latitude','housing_median_age','population','median_income','ocean_proximity_INLAND','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN','rooms_per_household','bedrooms_per_room','population_per_household']
threshold=10
vifObject=VIF(standardizedXTrain,threshold,columnsForCorrelation)
vifColumnsGreaterThanThreshold=vifObject.columnsWithGreaterVIFThreshold


model=Ridge()
modelFitObject = ModelFitting(model, standardizedXTrain, ytrain_)
fittedModel=modelFitObject.fittedModel


regressionMetrics_=mean_absolute_percentage_error 
evaluationObject=Evaluation(standardizedXTest,ytest_,regressionMetrics_,fittedModel)
score=evaluationObject.score

statModelOLSObject=StatModelOLS(standardizedXTrain,y)
print(statModelOLSObject.summary)

