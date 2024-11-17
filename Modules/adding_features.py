class AddingFeatures:
    def __init__(self,data,numerator,denominator,output):
        self.data=data
        self.numerator=numerator
        self.denominator=denominator
        self.output=output
        self.addingFeaturesDivision()
    def addingFeaturesDivision(self):
        self.data[self.output]=self.data[self.numerator]/self.data[self.denominator]
        
#cleanedData_['rooms_per_household']=cleanedData_['total_rooms']/cleanedData_['households']
# cleanedData_['bedrooms_per_room']=cleanedData_['total_bedrooms']/cleanedData_['total_rooms']
# cleanedData_['population_per_household']=cleanedData_['population']/cleanedData_['households']


