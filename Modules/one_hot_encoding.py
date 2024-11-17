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
        ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')
        ohetransform=ohe.fit_transform(self.data[self.categoricalColumns]).astype(int)
        data= pd.concat([self.data,ohetransform],axis=1).drop(columns=self.categoricalColumns)    
        return pd.DataFrame(data)
