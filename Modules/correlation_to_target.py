from scipy.stats import pearsonr, kendalltau
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class CorrelationToTarget:
    """
    Class to compute the correlation of all features in a dataset with a specified target variable
    and visualize it using a heatmap.
    """

    def __init__(self, columnsToCorrelateWith, data, target):
        """
        Initializes the CorrelationToTarget class with the provided dataset and target variable.

        Args:
            data (DataFrame): The input dataset containing features and the target variable.
            target (str): The name of the target variable for correlation analysis.
        """
        self.columnsToCorrelateWith = columnsToCorrelateWith
        self.target = target
        self.data = data
        self.colmnsToRelateWith = self.data[self.columnsToCorrelateWith].copy()
        self.correlationToTargetCorrMatrix = self.findingCorrelationToTargetCorrMatrix()
        self.pTestResults = self.pTest()
        self.kendallTauResults = self.kendallTau()

    def findingCorrelationToTargetCorrMatrix(self):
        """
        Computes the correlation matrix for the dataset and extracts correlations
        between all features and the target variable.

        Returns:
            Series: A sorted Series of correlation values between the target variable
                    and other features, in descending order.
        """
        corrMatrix = self.colmnsToRelateWith.corr()
        correlationToTarget = corrMatrix[self.target].sort_values(ascending=False)
        return correlationToTarget

    def pTest(self):
        """
        Performs Pearson correlation tests between each feature and the target variable.

        Returns:
            DataFrame: Correlation coefficients and p-values for numeric features.
        """
        results = {}
        for col in self.colmnsToRelateWith.select_dtypes(include=['number']).columns:
            if col != self.target:
                corr, p_value = pearsonr(self.colmnsToRelateWith[col], self.colmnsToRelateWith[self.target])
                results[col] = {'correlation': corr, 'p_value': p_value}
        return pd.DataFrame(results).T

    def kendallTau(self):
        """
        Performs Kendall's Tau correlation tests between each feature and the target variable.

        Returns:
            DataFrame: Correlation coefficients and Kendall Tau values for numeric features.
        """
        results = {}
        for col in self.colmnsToRelateWith.select_dtypes(include=['number']).columns:
            if col != self.target:
                corr, kendall_value = kendalltau(self.colmnsToRelateWith[col], self.colmnsToRelateWith[self.target])
                results[col] = {'correlation': corr, 'kendallTau': kendall_value}
        return pd.DataFrame(results).T

    def plot_heatmap(self):
        """
        Plots a heatmap of the correlation of all features with the target variable.
        """
        corrMatrix = self.colmnsToRelateWith.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corrMatrix[[self.target]].sort_values(by=self.target, ascending=False),
            annot=True,
            cmap="coolwarm",
            cbar=True
        )
        plt.title(f"Heatmap of Correlations with Target: {self.target}")
        plt.show()
