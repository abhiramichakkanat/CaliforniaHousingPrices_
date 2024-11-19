class CorrelationToEachOther:
    """
    Class to identify and handle highly correlated features in a dataset.

    Attributes:
        data (DataFrame): The input dataset for correlation analysis.
        correlatedPairs (list): A list of tuples containing pairs of highly correlated features.
        correlatedColumn (set): A set of feature names that are highly correlated with others (correlation > 0.9).
        withNoCorrelatedData (DataFrame): The dataset after removing highly correlated features.

    Methods:
        correlationToEachOther():
            Identifies highly correlated feature pairs and returns them along with a set of 
            correlated feature names.
        
        deleleCorrelatedColumn():
            Removes the highly correlated features from the dataset and returns the cleaned dataset.
    """
    def __init__(self,data):
        """
        Initializes the CorrelationToEachOther class with the provided dataset.

        Args:
            data (DataFrame): The input dataset to analyze for correlations.
        """
        self.data=data
        self.correlatedPairs,self.correlatedColumn=self.correlationToEachOther()
        self.withNoCorrelatedData=self.deleleCorrelatedColumn()
    def correlationToEachOther(self):
        """
        Identifies pairs of features in the dataset that have a high correlation (absolute value > 0.9).

        Returns:
            tuple:
                - list: A list of tuples containing pairs of highly correlated features.
                - set: A set of feature names that are highly correlated with others.
        """
        correlated_column=set()
        correlated_pairs=[]
        corr_matrix=self.data.corr()
        for i in range (len(corr_matrix.columns)):
                    for j in range (i):
                        if abs(corr_matrix.iloc[i,j]>0.9):
                            correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                            colname=corr_matrix.columns[i]
                            correlated_column.add(colname)
        return correlated_pairs,correlated_column
    
    def deleleCorrelatedColumn(self):
        """
        Removes highly correlated features from the dataset.

        Returns:
            DataFrame: The dataset after removing highly correlated features.
        """
        data = self.data.drop(self.correlatedColumn,axis=1)
        return data