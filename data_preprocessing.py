import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


riders_data = pd.read_csv('Data/Riders.csv')
train_data = pd.read_csv('Data/Train.csv')
test_data = pd.read_csv('Data/Test.csv')


def time_from_midnight_in_seconds(dataset, column):
    dataset[column] = pd.to_datetime(dataset[column])
    return (dataset[column] - pd.to_datetime(pd.to_datetime('today').date())).astype('timedelta64[s]')


def delta_time(dataset, higher_time, lower_time):
    return dataset[higher_time] - dataset[lower_time]


def average(list_data):
    return sum(list_data)/len(list_data)+1


def time_to_day_part(time):
    hours = time/3600
    if hours < 6:
        return 'night'
    if hours < 12:
        return 'morning'
    if hours < 18:
        return 'afternoon'
    else:
        return 'evening'


def calculate_bearing(lat1, lng1, lat2, lng2):
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    diffLong = np.deg2rad(lng2 - lng1)
    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1)
                                       * np.cos(lat2) * np.cos(diffLong))
    initial_bearing = np.arctan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = np.rad2deg(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


# converting time in seconds from midnight for train data
train_data['Placement - Time'] = time_from_midnight_in_seconds(train_data, 'Placement - Time')
train_data['Confirmation - Time'] = time_from_midnight_in_seconds(train_data, 'Confirmation - Time')
train_data['Pickup - Time'] = time_from_midnight_in_seconds(train_data, 'Pickup - Time')
train_data['Arrival at Pickup - Time'] = time_from_midnight_in_seconds(train_data, 'Arrival at Pickup - Time')
train_data['Arrival at Destination - Time'] = time_from_midnight_in_seconds(train_data, 'Arrival at Destination - Time')

# converting time in seconds from midnight for test data
test_data['Placement - Time'] = time_from_midnight_in_seconds(test_data, 'Placement - Time')
test_data['Confirmation - Time'] = time_from_midnight_in_seconds(test_data, 'Confirmation - Time')
test_data['Pickup - Time'] = time_from_midnight_in_seconds(test_data, 'Pickup - Time')
test_data['Arrival at Pickup - Time'] = time_from_midnight_in_seconds(test_data, 'Arrival at Pickup - Time')

# calculating delta_time for train data
delta_confirm_place_train = delta_time(train_data, 'Confirmation - Time', 'Placement - Time')
delta_pick_arr_confirm_train = delta_time(train_data, 'Arrival at Pickup - Time', 'Confirmation - Time')
delta_pickup_confirm_train = delta_time(train_data, 'Pickup - Time', 'Arrival at Pickup - Time')
delta_arrival_pickup_train = delta_time(train_data, 'Arrival at Destination - Time', 'Pickup - Time')
delta_placement_arrival_train = delta_time(train_data, 'Arrival at Destination - Time', 'Placement - Time')

# calculating delta_time for test data
delta_confirm_place_test = delta_time(test_data, 'Confirmation - Time', 'Placement - Time')
delta_pick_arr_confirm_test = delta_time(test_data, 'Arrival at Pickup - Time', 'Confirmation - Time')
delta_pickup_confirm_test = delta_time(test_data, 'Pickup - Time', 'Arrival at Pickup - Time')

# merging riders data with train data and test data
train_with_rider_info = train_data.merge(riders_data, on='Rider Id')
test_with_rider_info = test_data.merge(riders_data, on='Rider Id')

# adding Time from Pickup to Arrival column in test data
# test_with_rider_info['Time from Pickup to Arrival'] = delta_arrival_pickup_test

# drop outliers
train_with_rider_info = train_with_rider_info[train_with_rider_info['Placement - Weekday (Mo = 1)'] == train_with_rider_info['Confirmation - Weekday (Mo = 1)']]
train_with_rider_info = train_with_rider_info[train_with_rider_info['Placement - Day of Month'] == train_with_rider_info['Confirmation - Day of Month']]

# input missing values for Temperature  for train and test data
train_with_rider_info['Temperature'].fillna(train_with_rider_info['Temperature'].mean(), inplace=True)
test_with_rider_info['Temperature'].fillna(test_with_rider_info['Temperature'].mean(), inplace=True)

# label encoding of personal/business column for train data
labelencoder_personal_business = LabelEncoder()
train_with_rider_info['Personal or Business'] = labelencoder_personal_business.fit_transform(train_with_rider_info['Personal or Business'])

# label encoding of personal/business column for test data
labelencoder_personal_business = LabelEncoder()
test_with_rider_info['Personal or Business'] = labelencoder_personal_business.fit_transform(test_with_rider_info['Personal or Business'])


# one hot encoding of the train_data['Platform Type'] column
train_with_rider_info['Platform Type'] = train_with_rider_info['Platform Type'].astype('category')
train_with_rider_info = pd.concat([train_with_rider_info.drop(columns=['Platform Type']), pd.get_dummies(train_with_rider_info['Platform Type'])], axis=1)


# one hot encoding of the test_data['Platform Type'] column
test_with_rider_info['Platform Type'] = test_with_rider_info['Platform Type'].astype('category')
test_with_rider_info = pd.concat([test_with_rider_info.drop(columns=['Platform Type']), pd.get_dummies(test_with_rider_info['Platform Type'])], axis=1)

# drop redundant columns train data
train_with_rider_info.drop(columns=['Confirmation - Day of Month', 'Confirmation - Weekday (Mo = 1)',
                                    'Arrival at Pickup - Day of Month','Arrival at Pickup - Weekday (Mo = 1)',
                                    'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)',
                                    'Arrival at Destination - Day of Month',
                                    'Arrival at Destination - Weekday (Mo = 1)',
                                    'Vehicle Type', 'Order No',
                                    'User Id', 'Rider Id',
                                    'Precipitation in millimeters'], inplace=True)

# drop redundant columns test data
test_with_rider_info.drop(columns=['Confirmation - Day of Month', 'Confirmation - Weekday (Mo = 1)',
                                   'Arrival at Pickup - Day of Month', 'Arrival at Pickup - Weekday (Mo = 1)',
                                   'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)',
                                   'Vehicle Type',
                                   'User Id', 'Rider Id',
                                   'Precipitation in millimeters'], inplace=True)


# renaming columns
train_with_rider_info.rename(columns={1: "Platform Type 1", 2: "Platform Type 2",
                                      3: "Platform Type 3", 4: "Platform Type 4"}, inplace=True)

test_with_rider_info.rename(columns={1: "Platform Type 1", 2: "Platform Type 2",
                                     3: "Platform Type 3", 4: "Platform Type 4"}, inplace=True)


train_with_rider_info.rename(columns={'Placement - Day of Month': 'Day of Month',
                                      'Placement - Weekday (Mo = 1)': 'Weekday (Mo = 1)'})

test_with_rider_info.rename(columns={'Placement - Day of Month': 'Day of Month',
                                     'Placement - Weekday (Mo = 1)': 'Weekday (Mo = 1)'})

X = train_with_rider_info.drop(columns='Time from Pickup to Arrival')
Y = train_with_rider_info['Time from Pickup to Arrival']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


regressor = RandomForestRegressor(n_estimators=180, max_depth=110,
                                  max_features=3, min_samples_leaf=3,
                                  min_samples_split=8,
                                  random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

grid_param = {
    'n_estimators': [150, 160, 170, 180],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
}

gd_sr = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=grid_param,
    scoring='neg_mean_absolute_error',
    cv=4,
    n_jobs=-1)

gd_sr.fit(X_train, y_train)

best_parameters = gd_sr.best_params_


final_predict = regressor.predict(test_with_rider_info.drop(columns='Order No'))
test_with_rider_info['Time from Pickup to Arrival'] = final_predict

submission = test_with_rider_info[['Order No','Time from Pickup to Arrival' ]]
submission['Time from Pickup to Arrival'] = submission['Time from Pickup to Arrival'].astype(int)
submission.to_csv('/Users/rade_dragosavac/PycharmProjects/SendyLogisticsChallenge /Data/Submission.csv', index=False)








