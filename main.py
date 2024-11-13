import pandas as pd 
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_absolute_percentage_error,r2_score

from Modules.loading import Loading
from Modules.find_null_values import FindNullValues
from Modules.fillna import FillNa
from Modules.one_hot_encoding import OneHotEncoding
from Modules.outliers import Outlier
from Modules.splitting_data import SplittingData
from Modules.train_test_split import TrainTestSplit
from Modules.standardization import Standardization
from Modules.model_fitting import ModelFitting
from Modules.evaluation import Evaluation
from Modules.stat_models_ols import StatModelOLS

read_method_=pd.read_csv
path_=r"C:\Users\Admin\Desktop\Polestar_Work\Day-1_Preprocesing_2\Day-1_Preprocesing\housing\housing.csv"
loaderObject = Loading(path_,read_method_)
data_ = loaderObject.data

findNullValuesObject=FindNullValues(data_)
numericalNullColumns=findNullValuesObject.numericalNullColumns
categoricalNullColumns=findNullValuesObject.categoricalNullColumns


fillStrategy={'mean':[], 'drop':[],'median':[],'mode':[],'ffill':['total_bedrooms'],'bfill':[]}
fillNaObject = FillNa(data_, fillStrategy)
filledData_=fillNaObject.filledData


categoricalColumns_=["ocean_proximity"]
oneHotEncodedDataObject=OneHotEncoding(filledData_,categoricalColumns_)
oneHotEncodedData_=oneHotEncodedDataObject.oneHotEncodedData


method_outlier_="iqr"
Q1=0.25
Q2=0.75
multiplierLB=1.5
multiplierUB=1.5
outlierObject=Outlier(oneHotEncodedData_,method_outlier_,Q1,Q2,multiplierLB,multiplierUB)
cleanedData_=outlierObject.cleanedData
cleanedData_.info()

targetColumn_=['median_house_value']
splittingDataObject=SplittingData(cleanedData_,targetColumn_)
XData,yData=splittingDataObject.X,splittingDataObject.y
yData.info()

testSize=0.2
randomState=15
trainTestSplitObject=TrainTestSplit(XData, yData, testingSize=testSize, random_State=randomState)


xtrain_=trainTestSplitObject.xTrain
ytrain_=trainTestSplitObject.yTrain
xtest_=trainTestSplitObject.xTest
ytest_=trainTestSplitObject.yTest


method_ = 'yeo-johnson'
standardizationObject = Standardization(Xtrain=xtrain_, XTest=xtest_, method=method_)
standardizedXTrain=standardizationObject.xTrain_transformed
standardizedXTest=standardizationObject.xTest_transformed
standardizedXTrain.head()

model=Ridge()
modelFitObject = ModelFitting(model, standardizedXTrain, ytrain_)
fittedModel=modelFitObject.fittedModel
fittedModel

regressionMetrics_=mean_absolute_percentage_error 
evaluationObject=Evaluation(standardizedXTest,ytest_,regressionMetrics_,fittedModel)
evaluationObject.score

statModelOLSObject=StatModelOLS(XData,yData)
print(statModelOLSObject.summary)