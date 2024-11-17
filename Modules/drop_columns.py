import pandas as pd
class DropColumns:
    def __init__(self,data,columns):
        self.data=data
        self.columns=columns
        self.droppedData=self.deleteColumns()

    def deleteColumns(self):
        data = self.data.drop(self.columns,axis=1)
        
        return data
