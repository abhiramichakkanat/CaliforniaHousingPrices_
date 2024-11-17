import pandas as pd

class Evaluation():
    """
    Class to evaluate the fitted model using specified regression metrics.

    Attributes:
        xtest (DataFrame): Test features.
        ytest (Series): Test targets.
        regressionMetrics (function): Regression metric function.
        fittedModel (object): Fitted model.
    """    
    def __init__(self, XTest,ytest,regressionMetrics,fittedModel):
        """
        Initializes the Evaluation class with test data and evaluation metrics.

        Args:
            XTest (DataFrame): Test features.
            ytest (Series): Test targets.
            regressionMetrics (function): Metric function.
            fittedModel (object): Fitted model.
        """
        self.xtest=XTest
        self.ytest=ytest
        self.regressionMetrics=regressionMetrics
        self.fittedModel=fittedModel
        self.score=self.metrics()
    
    def metrics(self):
        """
        Evaluates the model performance using the provided regression metrics.

        Returns:
            float: Model performance score.
        """
        xtest=pd.DataFrame(self.xtest)
        yPred=self.fittedModel.predict(xtest)
        score=self.regressionMetrics(self.ytest,yPred)
        return score
