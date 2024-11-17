import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class OneHotEncoding:
    """
    Class to apply OneHotEncoding on categorical columns of the data.

    Attributes:
        data (DataFrame): The data to be encoded.
        categoricalColumns (list): The categorical columns to encode.
        oneHotEncodedData (DataFrame): The data after encoding.
    """
    def __init__(self,data,categoricalColumns):
        """
        Initializes the OneHotEncoding class with data and categorical columns.

        Args:
            data (DataFrame): The data to encode.
            categoricalColumns (list): List of columns to apply one-hot encoding.
        """
        self.data=data
        self.categoricalColumns=categoricalColumns
        self.oneHotEncodedData=self.oneHotEncoding()

    def oneHotEncoding(self):
        """
        Performs OneHotEncoding on the specified categorical columns.

        Returns:
            DataFrame: Data after one-hot encoding.
        """
        df_encoded = pd.get_dummies(self.data, columns=self.categoricalColumns,drop_first=False, dtype=float)
        return df_encoded
