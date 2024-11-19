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
        Detects and removes outliers from numeric columns in the dataset using the IQR method.

        Steps:
            1. Identify numeric columns in the dataset.
            2. For each numeric column:
               - Calculate Q1, Q3, and IQR.
               - Determine the lower and upper bounds based on multipliers.
               - Filter out rows with values outside the bounds.
            3. If the dataset becomes empty during the process, stop further processing.

        Returns:
            DataFrame: A cleaned dataset with outliers removed.
        """
        data=pd.DataFrame(self.oneHotEncodedData)
        
        cleaned_df = data.copy()
        

        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns

        for col in numeric_cols:
           
            if cleaned_df[col].nunique() <= 1:
                continue

            Q1 = cleaned_df[col].quantile(self.Q1)
            Q3 = cleaned_df[col].quantile(self.Q2)
            IQR = Q3 - Q1

            
            if IQR == 0:
                continue

            lower_bound = Q1 - self.multiplierLB * IQR
            upper_bound = Q3 + self.multiplierUB * IQR

            
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]

            
            if cleaned_df.empty:
               
                break

        return cleaned_df