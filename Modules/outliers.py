import pandas as pd
class Outlier:
    """
    Class to detect and remove outliers using specified method (IQR).

    Attributes:
        oneHotEncodedData (DataFrame): Data to clean.
        method (str): The method to use for outlier detection.
        cleanedData (DataFrame): The data after outlier removal.
    """
    def __init__(self,oneHotEncodedData,method,Q1,Q2,multiplierLB,multiplierUB):
        """
        Initializes the Outlier class with data and outlier detection method.

        Args:
            oneHotEncodedData (DataFrame): The data to detect and remove outliers.
            method (str): The method to use for outlier detection (e.g., 'iqr').
        """
        self.oneHotEncodedData=oneHotEncodedData
        self.method=method
        self.Q1=Q1
        self.Q2=Q2
        self.multiplierLB=multiplierLB
        self.multiplierUB=multiplierUB
        if(self.method=="iqr"):
            self.cleanedData=self.iqrOutlierDetectionAndRemoval()

    def iqrOutlierDetectionAndRemoval(self):
        """
        Removes outliers using IQR method.

        Returns:
            DataFrame: Data after outlier removal.
        """
        data=pd.DataFrame(self.oneHotEncodedData)
        dffeatureNames=data.columns
        cleaned_df = data.copy()
        for col in dffeatureNames:
                Q1 = cleaned_df[col].quantile(self.Q1)
                Q3 = cleaned_df[col].quantile(self.Q2)
                IQR = Q3 - Q1        
                lower_bound = Q1 - self.multiplierLB * IQR
                upper_bound = Q3 + self.multiplierUB * IQR
                cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]

        cleaned_df = cleaned_df.reset_index(drop=True)
        return pd.DataFrame(cleaned_df)