from scipy.stats import pearsonr, kendalltau
import pandas as pd
class CorrelationToTarget:
    """
    Class to compute the correlation of all features in a dataset with a specified target variable.

    Attributes:
        data (DataFrame): The input dataset containing features and the target variable.
        target (str): The name of the target variable for which correlations will be computed.
        correlationToTarget (Series): A sorted Series of correlation values between the target 
                                       variable and other features, in descending order.

    Methods:
        findingCorrelationToTarget():
            Calculates the correlation of all features in the dataset with the target variable 
            and sorts the results in descending order.
    """
    def __init__(self,columnsToCorrelateWith,data,target):
        """
        Initializes the CorrelationToTarget class with the provided dataset and target variable.

        Args:
            data (DataFrame): The input dataset containing features and the target variable.
            target (str): The name of the target variable for correlation analysis.
        """
        self.columnsToCorrelateWith=columnsToCorrelateWith
        self.target=target
        self.data=data
        self.colmnsToRelateWith = pd.DataFrame()
        self.colmnsToRelateWith = self.data[self.columnsToCorrelateWith].copy()
        self.correlationToTargetCorrMatrix=self.findingCorrelationToTargetCorrMatrix()
        self.pTestResults=self.pTest()
        self.kendallTauResults=self.kendallTau()

    def findingCorrelationToTargetCorrMatrix(self):
        """
        Computes the correlation matrix for the dataset and extracts correlations 
        between all features and the target variable.

        Returns:
            Series: A sorted Series of correlation values between the target variable 
                    and other features, in descending order.
        """
        
        corrMatrix=self.colmnsToRelateWith.corr()
        correlationToTarget= corrMatrix[self.target].sort_values(ascending=False)
        return correlationToTarget
         
    def pTest(self):
        results = {}
        for col in self.colmnsToRelateWith.select_dtypes(include=['number']).columns:
                if col != self.target:
                    corr, p_value = pearsonr(self.colmnsToRelateWith[col], self.colmnsToRelateWith[self.target])
                    results[col] = {'correlation': corr, 'p_value': p_value}
        return pd.DataFrame(results).T        
    
    def kendallTau(self):
        results = {}
        for col in self.colmnsToRelateWith.select_dtypes(include=['number']).columns:
                if col != self.target:
                    corr, kendalTlauValue = kendalltau(self.colmnsToRelateWith[col], self.colmnsToRelateWith[self.target])
                    results[col] = {'correlation': corr, 'kendallTau': kendalTlauValue}
        return pd.DataFrame(results).T      

