data_flights = pd.read_csv('modifiedFlights.csv',header=0,low_memory=False)
data_flights.head()
#Cleaning
#Finally, I clean the dataframe throwing the variables I won't use and re-organize the columns to ease its reading:
variables_to_remove = ['TAXI_OUT', 'TAXI_IN', 'WHEELS_ON', 'WHEELS_OFF', 'YEAR',
                       'MONTH','DAY','DAY_OF_WEEK','SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY','SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
        'SCHEDULED_TIME', 'ELAPSED_TIME','WEATHER_DELAY', 'DIVERTED','FLIGHT_NUMBER', 'TAIL_NUMBER', 'AIR_TIME']
data_flights.drop(variables_to_remove, axis = 1, inplace = True)
data_flights = data_flights[['CANCELLED','CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY','AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']]

#handling missing
missing_df = data_flights.isnull().sum(axis=0).reset_index()
missing_df.columns = ['variable', 'missing values']
missing_df['filling factor (%)']=(data_flights.shape[0]-missing_df['missing values'])/data_flights.shape[0]*100
missing_df.sort_values('filling factor (%)').reset_index(drop = True)
modifiedFLights.to_csv('modifiedFlights3.csv',index=False)
