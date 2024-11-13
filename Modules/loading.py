import pandas as pd
class Loading:
    """ 
    Class to load data from a specified path using a given read method.
    
    Attributes:
        path (str): The path to the data file.
        read_method (function): The method used to read the data.
        data (DataFrame): The loaded data.
    """
    def __init__(self, path_,read_method_):
        """ 
        Initializes the Loading class with a file path and read method.

        Args:
            path_ (str): The file path to the data.
            read_method_ (function): The method used to read the data.
        """
        self.path=path_
        self.read_method=read_method_
        self.data=self.getData()
    
    def getData(self):
        """
        Loads the data from the specified path using the read method.

        Returns:
            DataFrame: The loaded data.
        """
        return self.read_method(self.path)
