import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

class VIF:
    def __init__(self,data,threshold):
        self.data=data
        self.threshold=threshold
        self.vifGreaterThan10=self.removeVif()

    def removeVif(self):
        vif = pd.DataFrame()
        vif['Variable'] = self.data.columns
        vif['VIF'] = [variance_inflation_factor(self.data.values, i) for i in range(self.data.shape[1])]
        greaterThan10=vif[vif['VIF'] > self.threshold]
        greaterThan10=greaterThan10.to_numpy()
        greaterThreshold=[]
        for i in greaterThan10:
            greaterThreshold.append(i[0])
        return greaterThreshold
