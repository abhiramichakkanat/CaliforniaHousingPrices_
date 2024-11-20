import pandas as pd
from scipy.stats import pearsonr,kendalltau

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
    def __init__(self, correlationColumns,data):
        """
        Initializes the CorrelationToEachOther class with the provided dataset.

        Args:
            data (DataFrame): The input dataset to analyze for correlations.
        """
        self.data = data
        self.correlationColumns=correlationColumns
        self.colmnsToRelateWith = pd.DataFrame()
        self.colmnsToRelateWith = self.data[self.correlationColumns].copy()
        self.correlatedPairs, self.correlatedColumn = self.correlationToEachOther()
        self.pTestResultsofIndependentPairs = self.pTest()
        self.kendalTauResultsOfIndependentPairs= self.kendallTau()

    def correlationToEachOther(self):
        """
        Identifies pairs of features in the dataset that have a high correlation (absolute value > 0.9).

        Returns:
            tuple:
                - list: A list of tuples containing pairs of highly correlated features.
                - set: A set of feature names that are highly correlated with others.
        """
        correlated_column = set()
        correlated_pairs = []
        corr_matrix = self.colmnsToRelateWith.corr()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.9:  # Fixed condition
                    correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                    colname = corr_matrix.columns[i]
                    correlated_column.add(colname)
        return correlated_pairs, correlated_column

    def pTest(self):
        """
        Performs Pearson correlation tests between numeric columns and records correlation
        coefficients and p-values.

        Returns:
            DataFrame: A DataFrame of correlation coefficients and p-values for numeric column pairs.
        """
        results = []
        numeric_cols = self.colmnsToRelateWith.select_dtypes(include=['number']).columns

        for i in range(len(numeric_cols)):
            for j in range(0, len(numeric_cols)):  # Avoid duplicate pairs
                corr, p_value = pearsonr(self.colmnsToRelateWith[numeric_cols[i]], self.colmnsToRelateWith[numeric_cols[j]])
                results.append({
                    'Column1': numeric_cols[i],
                    'Column2': numeric_cols[j],
                    'Correlation': corr,
                    'P-Value': p_value
                })

        return pd.DataFrame(results)

    def kendallTau(self):
        """
        Performs Pearson correlation tests between numeric columns and records correlation
        coefficients and p-values.

        Returns:
            DataFrame: A DataFrame of correlation coefficients and p-values for numeric column pairs.
        """
        results = []
        numeric_cols = self.colmnsToRelateWith.select_dtypes(include=['number']).columns

        for i in range(len(numeric_cols)):
            for j in range(0, len(numeric_cols)):  # Avoid duplicate pairs
                corr, kendallValue = kendalltau(self.colmnsToRelateWith[numeric_cols[i]], self.colmnsToRelateWith[numeric_cols[j]])
                results.append({
                    'Column1': numeric_cols[i],
                    'Column2': numeric_cols[j],
                    'Correlation': corr,
                    'KendallTau-Value': kendallValue
                })

        return pd.DataFrame(results)