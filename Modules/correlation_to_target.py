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
    def __init__(self,data,target):
        """
        Initializes the CorrelationToTarget class with the provided dataset and target variable.

        Args:
            data (DataFrame): The input dataset containing features and the target variable.
            target (str): The name of the target variable for correlation analysis.
        """
        self.data=data
        self.target=target
        self.correlationToTarget=self.findingCorrelationToTarget()

    def findingCorrelationToTarget(self):
        """
        Computes the correlation matrix for the dataset and extracts correlations 
        between all features and the target variable.

        Returns:
            Series: A sorted Series of correlation values between the target variable 
                    and other features, in descending order.
        """
        corr_matrix=self.data.corr()
        correlationToTarget= corr_matrix[self.target].sort_values(ascending=False)
        return correlationToTarget
         

