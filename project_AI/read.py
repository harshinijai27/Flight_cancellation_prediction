import pandas as pd
flights = pd.read_csv('FLIGHTSsam.csv')
flights.isnull()
flights.isnull().sum()
flights['DEPARTURE_TIME'].isnull().sum()
modifiedFLights=flights.fillna("0")
#modifiedFLights.isnull.sum()
modifiedFLights.to_csv('modifiedFlights3.csv',index=False)
