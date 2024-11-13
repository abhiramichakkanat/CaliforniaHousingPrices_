import statsmodels.api as sm

class StatModelOLS:
    """
    Class to generate the summary of an Ordinary Least Squares regression model.

    Attributes:
        X (DataFrame): Feature data.
        y (Series): Target data.
        summary (str): Model summary.
    """
    def __init__(self,xData,ydata):
        """
        Initializes the StatModelOLS class with data.

        Args:
            xData (DataFrame): Feature data.
            ydata (Series): Target data.
        """
        self.X=xData
        self.y=ydata
        self.summary=self.statModel()

    def statModel(self):
        """
        Generates OLS summary for the provided data.

        Returns:
            str: Summary of the OLS model.
        """
        X = sm.add_constant(self.X)
        model = sm.OLS(self.y, X).fit()
        summary = model.summary()
        return summary
