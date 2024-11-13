class SplittingData:
    """
    Class to split the dataset into features and target variable.

    Attributes:
        data (DataFrame): The data to split.
        targetColumn (str): The target column for prediction.
        X (DataFrame): Feature data.
        y (Series): Target data.
    """
    def __init__(self,data,targetColumn):
        """
        Initializes the SplittingData class with data and target column.

        Args:
            data (DataFrame): The dataset.
            targetColumn (str): The target column name.
        """
        self.data=data
        self.targetColumn=targetColumn
        self.X,self.y=self.splitData()
        
    def splitData(self):
        """
        Splits data into features (X) and target variable (y).

        Returns:
            Tuple[DataFrame, Series]: Feature and target data.
        """
        X = self.data.drop(columns=self.targetColumn)  
        y = self.data[self.targetColumn]
        return X,y