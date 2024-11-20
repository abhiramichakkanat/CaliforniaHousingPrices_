import pandas as pd
from scipy.stats import zscore
import numpy as np
class Outlier:
    """
    Class to detect and remove outliers using specified method (IQR).

    Attributes:
        oneHotEncodedData (DataFrame): Data to clean.
        method (str): The method to use for outlier detection.
        cleanedData (DataFrame): The data after outlier removal.
    """
    def __init__(self,oneHotEncodedData,method,**kwargs):
        """
        Initializes the Outlier class with data and outlier detection method.

        Args:
            oneHotEncodedData (DataFrame): The data to detect and remove outliers.
            method (str): The method to use for outlier detection (e.g., 'iqr').
        """
        self.data=oneHotEncodedData
        self.method=method
        if(self.method=="iqr"):
            self.cleanedData=self.iqrOutlierDetectionAndRemoval(**kwargs)
        if(self.method=="zscore"):
            self.cleanedData=self.zscoreOutlierDetection(**kwargs)


    def zscoreOutlierDetection(self, **kwargs):
        """
    Detect outliers in the dataframe based on Z-Score threshold.
    Returns a dataframe with a boolean mask indicating outliers.
    
    :param df: DataFrame to check for outliers
    :param threshold: Z-score value above which a point is considered an outlier
    :return: A dataframe with True/False indicating outliers
    """
        threshold=kwargs.get('zscoreThreshold',3)
        z_scores = np.abs(zscore(self.data.select_dtypes(include=[np.number])))
        outliers_mask = (z_scores > threshold)
        cleaned_df = self.data[~outliers_mask.any(axis=1)].reset_index(drop=True)    
        return cleaned_df

    def iqrOutlierDetectionAndRemoval(self,**kwargs):
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
        
        quantile_lower=kwargs.get('quantile_lower',0.25)
        quantile_upper=kwargs.get('quantile_upper',0.75)
        multiplierLB=kwargs.get('multiplierLb',1.5)
        multiplierUB=kwargs.get('multiplierUb',1.5)
        data=pd.DataFrame(self.data)
        
        cleaned_df = data.copy()
        
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
           
            if cleaned_df[col].nunique() <= 1:
                continue
            Q1 = cleaned_df[col].quantile(quantile_lower)
            Q3 = cleaned_df[col].quantile(quantile_upper)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower_bound = Q1 - multiplierLB * IQR
            upper_bound = Q3 + multiplierUB * IQR            
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
            if cleaned_df.empty:
                break
        return cleaned_df