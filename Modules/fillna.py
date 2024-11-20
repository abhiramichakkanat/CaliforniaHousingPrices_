import pandas as pd
class FillNa:
    '''
    A class to handle missing value imputation strategies on a given dataset.
    
    Attributes:
    data : DataFrame
        The input dataset with missing values.
    fillStrategy : dict
        A dictionary where the key is the imputation strategy ('mean', 'median', 'mode', 'drop', 'ffill', 'bfill') and the value is a list of columns to which the strategy should be applied.
    filledData : DataFrame
        The dataset after missing values have been filled based on the provided strategies.

    Methods:
    apply() :
        Applies the fill strategies to the dataset and returns the modified dataset.
    fillMean(columns, data) :
        Fills missing values in specified columns with the mean of the column.
    fillMedian(columns, data) :
        Fills missing values in specified columns with the median of the column.
    fillMode(columns, data) :
        Fills missing values in specified columns with the mode (most frequent value) of the column.
    dropColumn(columns, data) :
        Drops the specified columns from the dataset if missing values are present.
    ffillColumn(columns, data) :
        Performs forward fill to fill missing values in specified columns.
    bfillColumn(columns, data) :
        Performs backward fill to fill missing values in specified columns.
    '''

    def __init__(self, data, fillStrategy,columns,**kwargs):
        '''
        Initializes the FillNa object with data and fill strategy.
        
        Parameters:
        data : DataFrame
            The dataset containing missing values.
        fillStrategy : dict
            A dictionary specifying the strategies and the columns to apply them to.
        '''
        self.data = data
        self.fillStrategy = fillStrategy.split()
        self.columns=columns.split()
        self.filledData = self.apply(**kwargs)

    def apply(self,**kwargs):
        '''
        Applies the fill strategies defined in the fillStrategy attribute to the data.
        
        Iterates through each strategy in fillStrategy and calls the corresponding method to fill missing values.
        
        Returns:
        DataFrame
            The dataset with missing values filled based on the provided strategies.
        '''
        
        filleddata = self.data.copy()
        for i in range(0,len(self.columns)):
            if self.fillStrategy[i] == 'mean':
                self.fillMean(self.columns[i], filleddata)
            elif self.fillStrategy[i] == 'median':
                self.fillMedian(self.columns[i], filleddata)
            elif self.fillStrategy[i] == 'mode':
                self.fillMode(self.columns[i],filleddata)
            elif self.fillStrategy[i] == 'drop':
                self.dropColumn(self.columns[i],filleddata)
            elif self.fillStrategy[i] == 'ffill':
                self.ffillColumn(self.columns[i],filleddata)
            elif self.fillStrategy[i] == 'bfill':
                self.bfillColumn(self.columns[i],filleddata)
            elif self.fillStrategy[i] == 'interpolate':
                self.interpolation(self.columns[i],filleddata)
            elif self.fillStrategy[i] =='movingAverage':
                self.fillMovingAverage(self.columns[i],filleddata,**kwargs)
        return filleddata   

    def interpolation(self,column,data):
        data[column].interpolate(inplace=True)
            

    def fillMovingAverage(self, column, data,**kwargs):
        '''
        Fills missing values in the specified columns with the moving average of the column.
        
        Parameters:
        columns : list
            A list of column names where missing values should be filled with the moving average.
        data : DataFrame
            The dataset in which missing values will be filled using moving averages.
        '''
        window=kwargs.get('window',10)
        if data[column].dtype in ['float64', 'int64']:  
            data[column] = data[column].fillna(data[column].rolling(window=window, min_periods=1).mean())



    def fillMean(self, columns, data):
        '''
        Fills missing values in the specified columns with the mean of the column.
        
        Parameters:
        columns : list
            A list of column names where missing values should be filled with the mean.
        data : DataFrame
            The dataset in which missing values will be filled.
        '''
        mean = data[columns].mean()
        data[columns].fillna(mean,inplace=True)

    def fillMedian(self, columns, data):
        '''
        Fills missing values in the specified columns with the median of the column.
        
        Parameters:
        columns : list
            A list of column names where missing values should be filled with the median.
        data : DataFrame
            The dataset in which missing values will be filled.
        '''
        median = data[columns].median()
        data[columns].fillna(median,inplace=True)
    
    def fillMode(self, columns, data):
        '''
        Fills missing values in the specified columns with the mode (most frequent value) of the column.
        
        Parameters:
        columns : list
            A list of column names where missing values should be filled with the mode.
        data : DataFrame
            The dataset in which missing values will be filled.
        '''

        mode = data[columns].mode()
        data[columns].fillna(mode,inplace=True)

    def dropColumn(self, columns, data):
        '''
        Drops the specified columns from the dataset if missing values are present.
        
        Parameters:
        columns : list
            A list of column names to drop from the dataset.
        data : DataFrame
            The dataset from which columns will be dropped.
        '''
        data.drop(columns, axis=1,inplace=True)

    def ffillColumn(self, columns, data):
        '''
        Performs forward fill to fill missing values in the specified columns.
        
        Parameters:
        columns : list
            A list of column names where missing values should be filled using forward fill.
        data : DataFrame
            The dataset in which missing values will be forward filled.
        '''
        
        data[columns].ffill(inplace=True)

    def bfillColumn(self, columns, data):
        '''
        Performs backward fill to fill missing values in the specified columns.
        
        Parameters:
        columns : list
            A list of column names where missing values should be filled using backward fill.
        data : DataFrame
            The dataset in which missing values will be backward filled.
        '''
        data[columns].bfill(inplace=True)