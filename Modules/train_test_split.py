from sklearn.model_selection import train_test_split

class TrainTestSplit:
    """
    Class to split data into training and testing sets.

    Attributes:
        X (DataFrame): Feature data.
        y (Series): Target data.
        testingSize (float): Test size proportion.
        randomState (int): Random seed for reproducibility.
    """
    def __init__(self, X,y,**kwargs):
        """
        Initializes the TrainTestSplit class with data and split parameters.

        Args:
            X (DataFrame): Feature data.
            y (Series): Target data.
            testingSize (float): Proportion of test data.
            randomState (int): Seed for reproducibility.
        """
        self.X=X
        self.y=y
        self.testingSize=kwargs.get('testingSize',0.2)
        self.randomState=kwargs.get('random_State',25)
        self.xTrain, self.xTest, self.yTrain, self.yTest = self.trainTestSplit()
        
    def trainTestSplit(self):
        """
        Splits data into training and testing sets based on specified parameters.

        Returns:
            Tuple: xTrain, xTest, yTrain, yTest.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.testingSize, random_state=self.randomState)
        return X_train, X_test, y_train, y_test
