class CorrelationToEachOther:
    def __init__(self,data):
        self.data=data
        self.correlatedPairs,self.correlatedColumn=self.correlationToEachOther()
        self.withNoCorrelatedData=self.deleleCorrelatedColumn()
    def correlationToEachOther(self):
        correlated_column=set()
        correlated_pairs=[]
        corr_matrix=self.data.corr()
        for i in range (len(corr_matrix.columns)):
                    for j in range (i):
                        if abs(corr_matrix.iloc[i,j]>0.9):
                            correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                            colname=corr_matrix.columns[i]
                            correlated_column.add(colname)
        return correlated_pairs,correlated_column
    
    def deleleCorrelatedColumn(self):
        data = self.data.drop(self.correlatedColumn,axis=1)
        return data