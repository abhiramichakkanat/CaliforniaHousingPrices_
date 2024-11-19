from sklearn.preprocessing import PowerTransformer,RobustScaler,StandardScaler,MinMaxScaler
import numpy as np
import pandas as pd
class Standardization:
    """
    Class to apply standardization (PowerTransformation) on training and test data.

    Attributes:
        xTrain (DataFrame): Training data.
        xTest (DataFrame): Testing data.
        method (str): Transformation method.
    """
    def __init__(self, **kwargs): 
        """
        Initializes the Standardization class with data and transformation method.

        Args:
            Xtrain (DataFrame): Training data.
            XTest (DataFrame): Testing data.
            method (str): Transformation method, e.g., 'yeo-johnson'.
        """
        self.xTrain = kwargs.get('Xtrain',None)
        self.xTest = kwargs.get('XTest',None)
        self.method = kwargs.get('method',None)
        if(self.method=='yeo-johnson' or 'box-cox'):
            self.pt = None
            self.xTrain_transformed = self.powerTransformer()
            self.xTest_transformed = self.powerTransformTest()
        
        
    def powerTransformer(self):
        """
        Applies power transformation on training data.

        Returns:
            DataFrame: Transformed training data.
        """
        data = pd.DataFrame(self.xTrain).copy()
        not_one_hot_columns = [
            col for col in data.columns 
            if data[col].dtype in [np.int64, np.float64] and not set(data[col].dropna()) <= {0, 1, 0.0, 1}
        ]
        self.pt = PowerTransformer(method=self.method)
        data[not_one_hot_columns] = self.pt.fit_transform(data[not_one_hot_columns])
        return data

    def powerTransformTest(self):
        """
        Applies power transformation on test data.

        Returns:
            DataFrame: Transformed test data.
        """
        data = pd.DataFrame(self.xTest).copy()
        not_one_hot_columns = [
            col for col in data.columns 
            if data[col].dtype in [np.int64, np.float64] and not set(data[col].dropna()) <= {0, 1, 0.0, 1}
        ]
        data[not_one_hot_columns] = self.pt.transform(data[not_one_hot_columns])
        return data
