class ModelFitting():
    """
    Class to fit a specified model on the training data.

    Attributes:
        model (object): The model to fit.
        xtrain (DataFrame): Training data.
        ytrain (Series): Target data.
    """
    def __init__(self, model,xtrain,ytrain): 
        """
        Initializes the ModelFitting class with model and training data.

        Args:
            model (object): The model instance to fit.
            xtrain (DataFrame): Training data.
            ytrain (Series): Target data.
        """
        self.xtrain=xtrain
        self.model=model
        self.ytrain=ytrain   
        self.fittedModel=self.modelFit()

    def modelFit(self):
        """
        Fits the specified model on the training data.

        Returns:
            object: Fitted model.
        """
        fittedModel = self.model.fit(self.xtrain, self.ytrain)
        return fittedModel
