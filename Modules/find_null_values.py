class FindNullValues:
    '''
    A class to identify and separate columns with null values in a dataset into categorical and numerical columns.
    
    Attributes:
    data : DataFrame
        The input dataset to search for null values.
    nullColumns : list
        A list of columns that contain null values.
    categoricalNullColumns : list
        A list of categorical columns that contain null values.
    numericalNullColumns : list
        A list of numerical columns that contain null values.

    Methods:
    findNullColumns() :
        Finds and returns all columns in the dataset that contain null values.
    seperateNullColumns() :
        Separates the null value columns into categorical and numerical columns based on their data type.
    '''
    def __init__(self,data):
        '''
        Initializes the FindNullValues object with data and identifies columns with null values.
        
        Parameters:
        data : DataFrame
            The dataset to check for null values.
        '''
        self.data=data
        self.nullColumns=self.findNullColumns()
        self.categoricalNullColumns,self.numericalNullColumns=self.seperateNullColumns()


    def findNullColumns(self):
        '''
        Identifies columns that contain null values in the dataset.
        
        Returns:
        list
            A list of column names that contain null values.
        '''
        nullColumns=self.data.columns[self.data.isna().any()].tolist()
        return nullColumns
    
    def seperateNullColumns(self):
        '''
        Separates columns containing null values into categorical and numerical columns based on their data type.
        
        Returns:
        tuple
            A tuple containing two lists:
            - categoricalNullColumns: columns with categorical data type that contain null values
            - numericalNullColumns: columns with numerical data type that contain null values
        '''
        numericalColumns=[]
        categoricalColumns=[]

        for i in self.nullColumns:
            if self.data[i].dtype=='o' :
                categoricalColumns.append(i)
            if self.data[i].dtype== 'float64':
                numericalColumns.append(i)
        return categoricalColumns,numericalColumns