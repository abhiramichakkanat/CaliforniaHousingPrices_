import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

class VIF:
    '''
    This class calculates the Variance Inflation Factor (VIF) for the variables in a given dataset 
    and identifies variables with VIF values greater than a specified threshold.

    Attributes:
        data (pd.DataFrame): The input dataset containing the variables to calculate VIF for.
        threshold (float): The VIF threshold for identifying multicollinearity.
        vifGreaterThan10 (list): A list of variable names with VIF values exceeding the threshold.

    Methods:
        removeVif():
            Calculates the VIF for each variable in the dataset and returns the names of variables
            with VIF values greater than the specified threshold.
    '''
    def __init__(self,data,threshold,columnsForCorrelation):
        '''
        Initializes the VIF class with the provided dataset and threshold.

        Parameters:
            data (pd.DataFrame): The input dataset containing variables.
            threshold (float): The VIF threshold for identifying multicollinearity.
        '''
        self.data=data
        self.columnsForCorrelation=columnsForCorrelation
        self.dataToCorrelate=pd.DataFrame()
        self.dataToCorrelate=self.data[self.columnsForCorrelation].copy()
        self.threshold=threshold
        self.columnsWithGreaterVIFThreshold=self.vifGreaterThanThreshold()

    def vifGreaterThanThreshold(self):
        '''
        Calculates VIF values for all variables in the dataset and identifies variables 
        with VIF values exceeding the threshold.

        Returns:
            list: A list of variable names with VIF values greater than the threshold.
        '''
        vif = pd.DataFrame()
        vif['Variable'] = self.dataToCorrelate.columns
        vif['VIF'] = [variance_inflation_factor(self.dataToCorrelate.values, i) for i in range(self.dataToCorrelate.shape[1])]
        greaterThanThreshold=vif[vif['VIF'] > self.threshold]
        greaterThanThreshold=greaterThanThreshold.to_numpy()
        greaterThreshold=[]
        for i in greaterThanThreshold:
            greaterThreshold.append(i[0])
        return greaterThreshold
