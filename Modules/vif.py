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
    def __init__(self,data,threshold):
        '''
        Initializes the VIF class with the provided dataset and threshold.

        Parameters:
            data (pd.DataFrame): The input dataset containing variables.
            threshold (float): The VIF threshold for identifying multicollinearity.
        '''
        self.data=data
        self.threshold=threshold
        self.vifGreaterThan10=self.removeVif()

    def removeVif(self):
        '''
        Calculates VIF values for all variables in the dataset and identifies variables 
        with VIF values exceeding the threshold.

        Returns:
            list: A list of variable names with VIF values greater than the threshold.
        '''
        vif = pd.DataFrame()
        vif['Variable'] = self.data.columns
        vif['VIF'] = [variance_inflation_factor(self.data.values, i) for i in range(self.data.shape[1])]
        greaterThan10=vif[vif['VIF'] > self.threshold]
        greaterThan10=greaterThan10.to_numpy()
        greaterThreshold=[]
        for i in greaterThan10:
            greaterThreshold.append(i[0])
        return greaterThreshold
