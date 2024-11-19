import pandas as pd
class DropColumns:
    """
    Class to drop specified columns from a given dataset.

    Attributes:
        data (DataFrame): The input dataset from which columns will be dropped.
        columns (list): The list of column names to be removed from the dataset.
        droppedData (DataFrame): The dataset after the specified columns have been dropped.

    Methods:
        deleteColumns():
            Drops the specified columns from the dataset and returns the updated dataset.
    """
    def __init__(self,data,columns):
        """
        Initializes the DropColumns class with the provided dataset and column names.

        Args:
            data (DataFrame): The input dataset from which columns will be dropped.
            columns (list): The list of column names to be removed from the dataset.
        """
        self.data=data
        self.columns=columns
        self.droppedData=self.deleteColumns()

    def deleteColumns(self):
        """
        Drops the specified columns from the dataset.

        Returns:
            DataFrame: A new dataset with the specified columns removed.
        """
        data = self.data.drop(self.columns,axis=1)
        
        return data
