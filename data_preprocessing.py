import pandas as pd
from sklearn.preprocessing import LabelEncoder


riders_data = pd.read_csv('Data/Riders.csv')
train_data = pd.read_csv('Data/Train.csv')
test_data = pd.read_csv('Data/Test.csv')


def time_from_midnight_in_seconds(column):
    train_data[column] = pd.to_datetime(train_data[column])
    return (train_data[column] - pd.to_datetime(pd.to_datetime('today').date())).astype('timedelta64[s]')


def delta_time(dataset, higher_time, lower_time):
    return dataset[higher_time] - dataset[lower_time]


def average(list_data):
    return sum(list_data)/len(list_data)+1


# converting time in seconds from midnight
train_data['Placement - Time'] = time_from_midnight_in_seconds('Placement - Time')
train_data['Confirmation - Time'] = time_from_midnight_in_seconds('Confirmation - Time')
train_data['Pickup - Time'] = time_from_midnight_in_seconds('Pickup - Time')
train_data['Arrival at Pickup - Time'] = time_from_midnight_in_seconds('Arrival at Pickup - Time')
train_data['Arrival at Destination - Time'] = time_from_midnight_in_seconds('Arrival at Destination - Time')


# calculating delta_time
delta_confirm_place = delta_time(train_data, 'Confirmation - Time', 'Placement - Time')
delta_pick_arr_confirm = delta_time(train_data, 'Arrival at Pickup - Time', 'Confirmation - Time')
delta_pickup_confirm = delta_time(train_data, 'Pickup - Time', 'Arrival at Pickup - Time')
delta_arrival_pickup = delta_time(train_data, 'Arrival at Destination - Time', 'Pickup - Time')
delta_placement_arrival = delta_time(train_data, 'Arrival at Destination - Time', 'Placement - Time')

# merging riders data with train data
train_with_rider_info = train_data.merge(riders_data, on='Rider Id')

# drop outliers
train_with_rider_info = train_with_rider_info[train_with_rider_info['Placement - Weekday (Mo = 1)'] == train_with_rider_info['Confirmation - Weekday (Mo = 1)']]
train_with_rider_info = train_with_rider_info[train_with_rider_info['Placement - Day of Month'] == train_with_rider_info['Confirmation - Day of Month']]

# input missing values
train_with_rider_info['Temperature'].fillna(train_with_rider_info['Temperature'].mean(), inplace=True)

# label encoding of personal / business column
labelencoder_personal_business = LabelEncoder()
train_with_rider_info['Personal or Business'] = labelencoder_personal_business.fit_transform(train_with_rider_info['Personal or Business'])

# one hot encoding of the train_data['Personal or Business'] column
train_with_rider_info['Platform Type'] = train_with_rider_info['Platform Type'].astype('category')
train_with_rider_info = pd.concat([train_with_rider_info.drop(columns=['Platform Type']), pd.get_dummies(train_with_rider_info['Platform Type'])], axis=1)

# drop redundant columns
train_with_rider_info.drop(columns=['Confirmation - Day of Month', 'Confirmation - Weekday (Mo = 1)',
                                    'Arrival at Pickup - Day of Month','Arrival at Pickup - Weekday (Mo = 1)',
                                    'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)',
                                    'Arrival at Destination - Day of Month',
                                    'Arrival at Destination - Weekday (Mo = 1)',
                                    'Vehicle Type'], inplace=True)


# renaming columns
train_with_rider_info.rename(columns={1: "Platform Type 1", 2: "Platform Type 2",
                                      3: "Platform Type 3", 4: "Platform Type 4"}, inplace=True)

train_with_rider_info.rename(columns={'Placement - Day of Month': 'Day of Month',
                                      'Placement - Weekday (Mo = 1)': 'Weekday (Mo = 1)'})
