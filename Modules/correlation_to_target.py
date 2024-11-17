class CorrelationToTarget:

    def __init__(self,data,target):
        self.data=data
        self.target=target
        self.correlationToTarget=self.findingCorrelationToTarget()

    def findingCorrelationToTarget(self):
        corr_matrix=self.data.corr()
        correlationToTarget= corr_matrix[self.target].sort_values(ascending=False)
        return correlationToTarget
         

