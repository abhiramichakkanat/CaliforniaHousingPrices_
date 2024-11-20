import pandas as pd
from scipy.stats import pearsonr,kendalltau

import seaborn as sns

import matplotlib.pyplot as plt
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
    def __init__(self, correlationColumns,data,threshold):
        """
        Initializes the CorrelationToEachOther class with the provided dataset.

        Args:
            data (DataFrame): The input dataset to analyze for correlations.
        """
        self.data = data
        self.threshold=threshold
        self.correlationColumns=correlationColumns
        self.colmnsToRelateWith = pd.DataFrame()
        self.colmnsToRelateWith = self.data[self.correlationColumns].copy()
        self.correlatedPairs, self.correlatedColumn = self.correlationToEachOther()
        self.pTestResultsofIndependentPairs = self.pTest()
        self.kendalTauResultsOfIndependentPairs= self.kendallTau()

    def correlationToEachOther(self):
        """
        Identifies pairs of features in the dataset that have a high correlation (absolute value > threshold).
        """
        correlated_column = set()
        correlated_pairs = []
        corr_matrix = self.colmnsToRelateWith.corr()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                    correlated_column.add(corr_matrix.columns[i])

        return correlated_pairs, correlated_column

    def plotHeatmap(self):
        """
        Plots a heatmap of the correlation matrix for visual analysis.
        """
        corr_matrix = self.colmnsToRelateWith.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
        plt.title("Correlation Matrix Heatmap")
        plt.show()

    def pTest(self):
        """
        Performs Pearson correlation tests between numeric columns, avoiding duplicate pairs and self-comparisons.

        Returns:
            DataFrame: A DataFrame of correlation coefficients and p-values for numeric column pairs.
        """
        results = []
        numeric_cols = self.colmnsToRelateWith.select_dtypes(include=['number']).columns

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):  # Ensure no duplicate pairs or self-comparisons
                col1, col2 = numeric_cols[i], numeric_cols[j]
                corr, p_value = pearsonr(self.colmnsToRelateWith[col1], self.colmnsToRelateWith[col2])
                results.append({
                    'Column1': col1,
                    'Column2': col2,
                    'Correlation': corr,
                    'P-Value': p_value
                })

        return pd.DataFrame(results)


    def kendallTau(self):
        """
        Performs Kendall's Tau correlation tests between numeric columns, avoiding duplicate pairs and self-comparisons.

        Returns:
            DataFrame: A DataFrame of correlation coefficients and Kendall Tau values for numeric column pairs.
        """
        results = []
        numeric_cols = self.colmnsToRelateWith.select_dtypes(include=['number']).columns

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):  # Ensure no duplicate pairs or self-comparisons
                col1, col2 = numeric_cols[i], numeric_cols[j]
                corr, kendall_value = kendalltau(self.colmnsToRelateWith[col1], self.colmnsToRelateWith[col2])
                results.append({
                    'Column1': col1,
                    'Column2': col2,
                    'Correlation': corr,
                    'KendallTau-Value': kendall_value
                })

        return pd.DataFrame(results)
