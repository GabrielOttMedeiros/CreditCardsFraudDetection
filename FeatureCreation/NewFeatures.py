## importing libraries
import pandas as pd
import numpy as np
import geopy.distance 
from dateutil.relativedelta import relativedelta
from datetime import datetime
import os

## Here we set the directory to look up the folder containing the data
os.chdir("/Users/gabrielmedeiros/Documents/OneDrive/Business analytics/DATA_445_Machine_Learning")


## Reading csv file, index_col = 0 makes the first column of the data to become the index of our pandas data frame
train_data = pd.read_csv('fraudTrain.csv', index_col = 0)
test_data = pd.read_csv('fraudTest.csv', index_col = 0)

#########################
## FEATURE ENGINEERING ##
#########################

## Here assign to n the amount of observations we have in the dataset
n = train_data.shape[0]

##############
## Distance ##
##############

## Creating empty list to store the reuslts
distance_to_append = []

## Looping through each row an computing the distance between transaction address and merchants address
for i in range(0,n):

    ## Here we gather the lat and long from the transaciton address
    coords_1 = (train_data['lat'][i], train_data['long'][i])

    ## Here we gather the lat and long from the merchants address
    coords_2 = (train_data['merch_lat'][i], train_data['merch_long'][i])

    ## Here we compute the disance in miles between the locations
    distance_to_append.append(geopy.distance.geodesic(coords_1, coords_2).miles)

## Adding results to our data set
train_data['distance'] = distance_to_append

#############################
## AVG Distance by Category##
#############################

## Here we create a groupby function to get the mean distance by each category
avg_dist_by_category = pd.DataFrame(train_data.groupby(['cc_num','category'])['distance'].mean())

## Here we create a temporary column holding our index values (cc_num)
avg_dist_by_category['columns'] = avg_dist_by_category.index

## Here we drop our temporary column and reset our index, which was previously our cc_num
avg_dist_by_category = avg_dist_by_category.reset_index().drop(columns = 'columns')

## Here we rename the columns of the groupby function
avg_dist_by_category.columns = ['cc_num','category','avg_distance_by_category']

## Here we merge our temporary data frame with our data set
train_data = avg_dist_by_category.merge(train_data, on = ['cc_num','category'], how = 'left')

##################
## AVG Distance ##
##################

## Here we create a groupby function to get the mean distance by each category
avg_dist = pd.DataFrame(train_data.groupby(['cc_num'])['distance'].mean())

## Here we create a temporary column holding our index values (cc_num)
avg_dist['columns'] = avg_dist.index

## Here we drop our temporary column and reset our index, which was previously our cc_num
avg_dist = avg_dist.reset_index().drop(columns = 'columns')

## Here we rename the columns of the groupby function
avg_dist.columns = ['cc_num','avg_distance']

## Here we merge our temporary data frame with our data set
train_data = avg_dist.merge(train_data, on = ['cc_num'], how = 'left')

#########
## Age ##
#########

## Creating empty list to store the resutls
ages_to_append = []

## Looping through each observation and computing the age of each individual from its DOB
for i in range(0,n):

    ## Here we add the last date of this year
    year_of_2021 = datetime.strptime('2021-12-31', "%Y-%m-%d")

    ## Here we call each DOB in the dataset
    dob = datetime.strptime(train_data.dob[i], "%Y-%m-%d")

    ## Here we compute the ages
    ages_to_append.append(relativedelta(year_of_2021, dob).years)

## Adding results to our data set
train_data['age'] = ages_to_append

######################
## Days of the Week ##
######################

## Here we change the format of our date column
dates = pd.to_datetime(train_data['trans_date_trans_time'])

## Here we use our library to get the day of the week based on the transformed column
train_data['day_of_week'] = dates.dt.day_name()

##################
## Uses per day ##
##################

## Here we create a dataframe containing the uses by day per card
uses_per_day = pd.DataFrame(train_data.groupby('cc_num')['day_of_week'].value_counts())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
uses_per_day['columns'] = uses_per_day.index

## Here we rename the columns
uses_per_day.columns = ['uses_per_day','columns']

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
uses_per_day = uses_per_day.reset_index().drop(columns = 'columns')

## Here we merge the new data to our table
train_data = uses_per_day.merge(train_data, on = ['cc_num', 'day_of_week'], how = 'left')

#######################
## Month of the year ##
#######################

## Here we change the format of our date column
months = pd.to_datetime(train_data['trans_date_trans_time'])

## Here we use our library to get the month of the year based on the transformed column
train_data['month'] = dates.dt.month

####################
## Uses per month ##
####################

## Here we create a dataframe containing the card uses by month
uses_per_month = pd.DataFrame(train_data.groupby('cc_num')['month'].value_counts())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
uses_per_month['columns'] = uses_per_month.index

## Here we rename the columns
uses_per_month.columns = ['uses_per_month','columns']

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
uses_per_month = uses_per_month.reset_index().drop(columns = 'columns')

## Here we merge the new data to our table
train_data = uses_per_month.merge(train_data, on = ['cc_num','month'], how = 'left')

#####################
## Hour of the day ##
#####################

## Here we change the format of our date column
hour_of_the_day = []

## Here we loop throough each observation, change the format of the items in each column, 
## and retrieve the hour of the day
for i in range(0,n):
    hour_of_the_day.append(datetime.strptime(train_data.trans_date_trans_time[i] ,"%Y-%m-%d %H:%M:%S").hour)

## Here we attribute our results to a column
train_data['hour_of_the_day'] = hour_of_the_day

###################
## Uses per hour ##
###################

## Here we create a dataframe containing the card uses by hour of day
uses_per_hour = pd.DataFrame(train_data.groupby('cc_num')['hour_of_the_day'].value_counts())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
uses_per_hour['columns'] = uses_per_hour.index

## Here we rename the columns
uses_per_hour.columns = ['uses_per_hour','columns']

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
uses_per_hour = uses_per_hour.reset_index().drop(columns = 'columns')

## Here we merge the new data to our table
train_data = uses_per_hour.merge(train_data, on = ['cc_num','hour_of_the_day'], how = 'left')

#################################
## Total card uses by customer ##
#################################

## Here we get the number of transactions shown in the data set per card
Uses = pd.DataFrame(train_data['cc_num'].value_counts())

## Here we create a column to keep our cc_num
Uses['cc_number2'] = Uses.index

## Here we reset our index
Uses = Uses.reset_index(drop = True)

## Here we rename our columns
Uses.columns = ['total_uses','cc_num']

## Here we merge our temporary data frame with our data set
train_data = Uses.merge(train_data, on = 'cc_num', how = 'left')

##########################################
## Total card uses by customer grouping ##
##########################################

conditions = [
    (train_data['total_uses'] < 400),
    (train_data['total_uses'] > 400) & (train_data['total_uses'] < 900),
    (train_data['total_uses'] > 900) & (train_data['total_uses'] < 1200),
    (train_data['total_uses'] > 1200) & (train_data['total_uses'] < 1800),
    (train_data['total_uses'] > 1800) & (train_data['total_uses'] < 2200),
    (train_data['total_uses'] > 2200) & (train_data['total_uses'] < 2800),
    (train_data['total_uses'] > 2800) & (train_data['total_uses'] < 3200)]

    
classes = [
    'LESS THAN 400',
    'BETWEEN 400 AND 900',
    'BETWEEN 900 AND 1200',
    'BETWEEN 1200 AND 1800',
    'BETWEEN 1800 AND 2200',
    'BETWEEN 2200 AND 2800',
    'BETWEEN 2800 AND 3200'
    ]
train_data['transactions_group'] = np.select(conditions,classes)

####################################################
## Difference in minutes between each transaction ##
####################################################

## Here we create a list to hold our results
new_time = []

## Here we loop through each observation and transform the format of our data
for i in range(0,n):
    new_time.append(datetime.strptime(train_data.trans_date_trans_time[i] ,"%Y-%m-%d %H:%M:%S"))

## Here we create a column containing transformed data to compute the difference in minutes
## New format was recquired for the library to work
train_data['transformed_time'] = new_time

train_data = train_data.sort_values(by = 'transformed_time', ascending = True)

## Here we compute the difference in minutes between each transacion
train_data['diff_by_card_trans'] = train_data.groupby('cc_num')\
                              ['transformed_time'].diff().apply(lambda x: \
                              x/np.timedelta64(1, 'm')).fillna(0).astype('int64')

#############################
## AVG payment by category ##
#############################

## Here we create a dataframe containing the avg amount per category
amt_per_category = pd.DataFrame(train_data.groupby(['cc_num','category'])['amt'].mean())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
amt_per_category['columns'] = amt_per_category.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
amt_per_category = amt_per_category.reset_index().drop(columns = 'columns')

## Here we rename the columns
amt_per_category.columns = ['cc_num','category','avg_by_category']

## Here we merge the new data to our table
train_data = amt_per_category.merge(train_data, on = ['cc_num','category'], how = 'left')

###############################
## AVG amount spent per card ##
###############################

## Here we create a dataframe containing the avg amount spent per card
avg_amt = pd.DataFrame(train_data.groupby('cc_num')['amt'].mean())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
avg_amt['columns'] = avg_amt.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
avg_amt = avg_amt.reset_index().drop(columns = 'columns')

## Here we rename the columns
avg_amt.columns = ['cc_num', 'avg_amt']

## Here we merge the new data to our table
train_data = avg_amt.merge(train_data, on = 'cc_num', how = 'left')

#################################################
## First and last purchase of each credit card ##
#################################################

## Here we extract the first and last transaction time recorded for each credit card
temp_cc_time_max = pd.DataFrame(train_data.groupby('cc_num')['trans_date_trans_time'].max())
temp_cc_time_min = pd.DataFrame(train_data.groupby('cc_num')['trans_date_trans_time'].min())

## Here we assign the index(cc_num) to a column
temp_cc_time_max['columns'] = temp_cc_time_max.index
temp_cc_time_min['columns'] = temp_cc_time_min.index

## Here we reset the index
temp_cc_time_max = temp_cc_time_max.reset_index().drop(columns = 'columns')
temp_cc_time_min = temp_cc_time_min.reset_index().drop(columns = 'columns')

## Here we rename the columns
temp_cc_time_max.columns = ['cc_num','max_date']
temp_cc_time_min.columns = ['cc_num','min_date']

## Here we merge the new columns to our original data
train_data = temp_cc_time_max.merge(train_data, on = 'cc_num', how = 'left')
train_data = temp_cc_time_min.merge(train_data, on = 'cc_num', how = 'left')

#############################################################################
## Diffefence in time between first and last purchase for each credit card ##
#############################################################################

## Here we transform the date type for each column
min_dates = pd.to_datetime(train_data.min_date) 
max_dates = pd.to_datetime(train_data.max_date)

## Here we compute the diff in minutes between first and last purchase
diff_in_time = max_dates - min_dates

## Here we create a dataframe with our results
temp_data = pd.DataFrame(diff_in_time)

## Here we add our results to a column in our data
train_data['diff_first_last'] = temp_data.apply(lambda x: x/np.timedelta64(1, 'm')).fillna(0).astype('int64')

#########################
## STD amount per card ##
#########################

## Here we create a dataframe containing the std from amount spent per card
amt_std = pd.DataFrame(train_data.groupby('cc_num')['amt'].std())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
amt_std['columns'] = amt_std.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
amt_std = amt_std.reset_index().drop(columns = 'columns')

## Here we rename the columns
amt_std.columns = ['cc_num', 'std_amt']

## Here we merge the new data to our table
train_data = amt_std.merge(train_data, on = 'cc_num', how = 'left')

#####################################
## STD amount per card by category ##
#####################################

## Here we create a dataframe containing the std from amount spent per card by category
amt_std = pd.DataFrame(train_data.groupby(['cc_num','category'])['amt'].std())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
amt_std['columns'] = amt_std.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
amt_std = amt_std.reset_index().drop(columns = 'columns')

## Here we rename the columns
amt_std.columns = ['cc_num','category', 'std_amt_by_category']

## Here we merge the new data to our table
train_data = amt_std.merge(train_data, on = ['cc_num','category'], how = 'left')

########################################
## STD distance per card by category ##
#######################################

## Here we create a dataframe containing the std distance per card by category
amt_std = pd.DataFrame(train_data.groupby(['cc_num','category'])['distance'].std())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
amt_std['columns'] = amt_std.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
amt_std = amt_std.reset_index().drop(columns = 'columns')

## Here we rename the columns
amt_std.columns = ['cc_num', 'category','std_dist_by_category']

## Here we merge the new data to our table
train_data = amt_std.merge(train_data, on = ['cc_num','category'], how = 'left')

###########################
## STD distance per card ##
###########################

## Here we create a dataframe containing the std distance per card by category
amt_std = pd.DataFrame(train_data.groupby('cc_num')['distance'].std())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
amt_std['columns'] = amt_std.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
amt_std = amt_std.reset_index().drop(columns = 'columns')

## Here we rename the columns
amt_std.columns = ['cc_num', 'std_dist']

## Here we merge the new data to our table
train_data = amt_std.merge(train_data, on = 'cc_num', how = 'left')

################################################
## Purhase amt and dist > avg and > (1,2) std ##
################################################

## Here we create a binary column containing 0 if amount spent is less than avg amount
train_data['purchase_>_avg'] = np.where(train_data['amt'] > train_data['avg_amt'],1,0)

## Here we create a binary column containing 0 if amount spent is less than avg amount per category
train_data['purchase_>_avg_by_category'] = np.where(train_data['amt'] > train_data['avg_by_category'],1,0)

## Here we create a binary column containing 0 if distance is less than avg distance
train_data['purchase_>_distance'] = np.where(train_data['distance'] > train_data['avg_distance'],1,0)

## Here we create a binary column containing 0 if distance is less than avg distance by category
train_data['purchase_>_distance_by_category'] = np.where(train_data['distance'] > train_data['avg_distance_by_category'],1,0)


## Here we classify if the amount spent is higher than 1 std
train_data['amt_>_1_std'] = np.where(train_data.amt > (train_data.avg_amt + train_data.std_amt),1,0)
## Here we classify if the amount spent is higher than 2 std
train_data['amt_>_2_std'] = np.where(train_data.amt > (train_data.avg_amt + 2*train_data.std_amt),1,0)


## Here we classify if the amount spent by category is higher than 1 std
train_data['amt_>_1_std_by_cagegory'] = np.where(train_data.amt > (train_data.avg_by_category + train_data.std_amt_by_category),1,0)
## Here we classify if the amount spent by category is higher than 2 std
train_data['amt_>_2_std_by_cagegory'] = np.where(train_data.amt > (train_data.avg_by_category + 2*train_data.std_amt_by_category),1,0)


## Here we classify if the distance is higher than 1 std
train_data['dist_>_1_std'] = np.where(train_data.distance > (train_data.avg_distance + train_data.std_dist),1,0)
## Here we classify if the distance is higher than 2 std
train_data['dist_>_2_std'] = np.where(train_data.distance > (train_data.avg_distance + 2*train_data.std_dist),1,0)


## Here we classify if the distance by category is higher than 1 std
train_data['dist_>_1_std_by_cagegory'] = np.where(train_data.distance > (train_data.avg_distance_by_category + train_data.std_dist_by_category),1,0)
## Here we classify if the dintance by category is higher than 2 std
train_data['dist_>_2_std_by_cagegory'] = np.where(train_data.distance > (train_data.avg_distance_by_category + 2*train_data.std_dist_by_category),1,0)




#######################
## Uses per category ##
#######################

## Here we create a dataframe containing the uses by category per card
uses_per_category = pd.DataFrame(train_data.groupby('cc_num')['category'].value_counts())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
uses_per_category['columns'] = uses_per_category.index

## Here we rename the columns
uses_per_category.columns = ['uses_per_category','columns']

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
uses_per_category = uses_per_category.reset_index().drop(columns = 'columns')

## Here we merge the new data to our table
train_data = uses_per_category.merge(train_data, on = ['cc_num','category'], how = 'left')


##############
## Distance ##
##############


## Here assign to n the amount of observations we have in the dataset
n = test_data.shape[0]

## Creating empty list to store the reuslts
distance_to_append = []

## Looping through each row an computing the distance between transaction address and merchants address
for i in range(0,n):

    ## Here we gather the lat and long from the transaciton address
    coords_1 = (test_data['lat'][i], test_data['long'][i])

    ## Here we gather the lat and long from the merchants address
    coords_2 = (test_data['merch_lat'][i], test_data['merch_long'][i])

    ## Here we compute the disance in miles between the locations
    distance_to_append.append(geopy.distance.geodesic(coords_1, coords_2).miles)

## Adding results to our data set
test_data['distance'] = distance_to_append


##################
## AVG Distance ##
##################

## Here we create a groupby function to get the mean distance by each category
avg_dist_by_category = pd.DataFrame(test_data.groupby(['cc_num','category'])['distance'].mean())

## Here we create a temporary column holding our index values (cc_num)
avg_dist_by_category['columns'] = avg_dist_by_category.index

## Here we drop our temporary column and reset our index, which was previously our cc_num
avg_dist_by_category = avg_dist_by_category.reset_index().drop(columns = 'columns')

## Here we rename the columns of the groupby function
avg_dist_by_category.columns = ['cc_num','category','avg_distance_by_category']

## Here we merge our temporary data frame with our data set
test_data = avg_dist_by_category.merge(test_data, on = ['cc_num','category'], how = 'left')

#############################
## AVG Distance by Category##
#############################

## Here we create a groupby function to get the mean distance by each category
avg_dist = pd.DataFrame(test_data.groupby(['cc_num'])['distance'].mean())

## Here we create a temporary column holding our index values (cc_num)
avg_dist['columns'] = avg_dist.index

## Here we drop our temporary column and reset our index, which was previously our cc_num
avg_dist = avg_dist.reset_index().drop(columns = 'columns')

## Here we rename the columns of the groupby function
avg_dist.columns = ['cc_num','avg_distance']

## Here we merge our temporary data frame with our data set
test_data = avg_dist.merge(test_data, on = ['cc_num'], how = 'left')

#########
## Age ##
#########

## Creating empty list to store the resutls
ages_to_append = []

## Looping through each observation and computing the age of each individual from its DOB
for i in range(0,n):

    ## Here we add the last date of this year
    year_of_2021 = datetime.strptime('2021-12-31', "%Y-%m-%d")

    ## Here we call each DOB in the dataset
    dob = datetime.strptime(test_data.dob[i], "%Y-%m-%d")

    ## Here we compute the ages
    ages_to_append.append(relativedelta(year_of_2021, dob).years)

## Adding results to our data set
test_data['age'] = ages_to_append

######################
## Days of the Week ##
######################

## Here we change the format of our date column
dates = pd.to_datetime(test_data['trans_date_trans_time'])

## Here we use our library to get the day of the week based on the transformed column
test_data['day_of_week'] = dates.dt.day_name()

##################
## Uses per day ##
##################

## Here we create a dataframe containing the uses by day per card
uses_per_day = pd.DataFrame(test_data.groupby('cc_num')['day_of_week'].value_counts())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
uses_per_day['columns'] = uses_per_day.index

## Here we rename the columns
uses_per_day.columns = ['uses_per_day','columns']

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
uses_per_day = uses_per_day.reset_index().drop(columns = 'columns')

## Here we merge the new data to our table
test_data = uses_per_day.merge(test_data, on = ['cc_num','day_of_week'], how = 'left')

#######################
## Month of the year ##
#######################

## Here we change the format of our date column
months = pd.to_datetime(test_data['trans_date_trans_time'])

## Here we use our library to get the month of the year based on the transformed column
test_data['month'] = dates.dt.month

####################
## Uses per month ##
####################

## Here we create a dataframe containing the card uses by month
uses_per_month = pd.DataFrame(test_data.groupby('cc_num')['month'].value_counts())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
uses_per_month['columns'] = uses_per_month.index

## Here we rename the columns
uses_per_month.columns = ['uses_per_month','columns']

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
uses_per_month = uses_per_month.reset_index().drop(columns = 'columns')

## Here we merge the new data to our table
test_data = uses_per_month.merge(test_data, on = ['cc_num','month'], how = 'left')

#####################
## Hour of the day ##
#####################

## Here we change the format of our date column
hour_of_the_day = []

## Here we loop throough each observation, change the format of the items in each column, 
## and retrieve the hour of the day
for i in range(0,n):
    hour_of_the_day.append(datetime.strptime(test_data.trans_date_trans_time[i] ,"%Y-%m-%d %H:%M:%S").hour)

## Here we attribute our results to a column
test_data['hour_of_the_day'] = hour_of_the_day

###################
## Uses per hour ##
###################

## Here we create a dataframe containing the card uses by hour of day
uses_per_hour = pd.DataFrame(test_data.groupby('cc_num')['hour_of_the_day'].value_counts())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
uses_per_hour['columns'] = uses_per_hour.index

## Here we rename the columns
uses_per_hour.columns = ['uses_per_hour','columns']

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
uses_per_hour = uses_per_hour.reset_index().drop(columns = 'columns')

## Here we merge the new data to our table
test_data = uses_per_hour.merge(test_data, on = ['cc_num','hour_of_the_day'], how = 'left')

#################################
## Total card uses by customer ##
#################################

## Here we get the number of transactions shown in the data set per card
Uses = pd.DataFrame(test_data['cc_num'].value_counts())

## Here we create a column to keep our cc_num
Uses['cc_number2'] = Uses.index

## Here we reset our index
Uses = Uses.reset_index(drop = True)

## Here we rename our columns
Uses.columns = ['total_uses','cc_num']

## Here we merge our temporary data frame with our data set
test_data = Uses.merge(test_data, on = 'cc_num', how = 'left')

##########################################
## Total card uses by customer grouping ##
##########################################

conditions = [
    (test_data['total_uses'] < 400),
    (test_data['total_uses'] > 400) & (test_data['total_uses'] < 900),
    (test_data['total_uses'] > 900) & (test_data['total_uses'] < 1200),
    (test_data['total_uses'] > 1200) & (test_data['total_uses'] < 1800),
    (test_data['total_uses'] > 1800) & (test_data['total_uses'] < 2200),
    (test_data['total_uses'] > 2200) & (test_data['total_uses'] < 2800),
    (test_data['total_uses'] > 2800) & (test_data['total_uses'] < 3200)
    ]

    
classes = [
    'LESS THAN 400',
    'BETWEEN 400 AND 900',
    'BETWEEN 900 AND 1200',
    'BETWEEN 1200 AND 1800',
    'BETWEEN 1800 AND 2200',
    'BETWEEN 2200 AND 2800',
    'BETWEEN 2800 AND 3200'
    ]
test_data['transactions_group'] = np.select(conditions,classes)

####################################################
## Difference in minutes between each transaction ##
####################################################

## Here we create a list to hold our results
new_time = []
n = test_data.shape[0]
## Here we loop through each observation and transform the format of our data
for i in range(0,n):
    new_time.append(datetime.strptime(test_data.trans_date_trans_time[i] ,"%Y-%m-%d %H:%M:%S"))

## Here we create a column containing transformed data to compute the difference in minutes
## New format was recquired for the library to work
test_data['transformed_time'] = new_time

test_data = test_data.sort_values(by = 'transformed_time', ascending = True)

## Here we compute the difference in minutes between each transacion
test_data['diff_by_card_trans'] = test_data.groupby('cc_num')\
                              ['transformed_time'].diff().apply(lambda x: \
                              x/np.timedelta64(1, 'm')).fillna(0).astype('int64')

#############################
## AVG payment by category ##
#############################

## Here we create a dataframe containing the avg amount per category
amt_per_category = pd.DataFrame(test_data.groupby(['cc_num','category'])['amt'].mean())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
amt_per_category['columns'] = amt_per_category.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
amt_per_category = amt_per_category.reset_index().drop(columns = 'columns')

## Here we rename the columns
amt_per_category.columns = ['cc_num','category','avg_by_category']

## Here we merge the new data to our table
test_data = amt_per_category.merge(test_data, on = ['cc_num','category'], how = 'left')

###############################
## AVG amount spent per card ##
###############################

## Here we create a dataframe containing the avg amount spent per card
avg_amt = pd.DataFrame(test_data.groupby('cc_num')['amt'].mean())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
avg_amt['columns'] = avg_amt.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
avg_amt = avg_amt.reset_index().drop(columns = 'columns')

## Here we rename the columns
avg_amt.columns = ['cc_num', 'avg_amt']

## Here we merge the new data to our table
test_data = avg_amt.merge(test_data, on = 'cc_num', how = 'left')

#################################################
## First and last purchase of each credit card ##
#################################################

## Here we extract the first and last transaction time recorded for each credit card
temp_cc_time_max = pd.DataFrame(test_data.groupby('cc_num')['trans_date_trans_time'].max())
temp_cc_time_min = pd.DataFrame(test_data.groupby('cc_num')['trans_date_trans_time'].min())

## Here we assign the index(cc_num) to a column
temp_cc_time_max['columns'] = temp_cc_time_max.index
temp_cc_time_min['columns'] = temp_cc_time_min.index

## Here we reset the index
temp_cc_time_max = temp_cc_time_max.reset_index().drop(columns = 'columns')
temp_cc_time_min = temp_cc_time_min.reset_index().drop(columns = 'columns')

## Here we rename the columns
temp_cc_time_max.columns = ['cc_num','max_date']
temp_cc_time_min.columns = ['cc_num','min_date']

## Here we merge the new columns to our original data
test_data = temp_cc_time_max.merge(test_data, on = 'cc_num', how = 'left')
test_data = temp_cc_time_min.merge(test_data, on = 'cc_num', how = 'left')

#############################################################################
## Diffefence in time between first and last purchase for each credit card ##
#############################################################################

## Here we transform the date type for each column
min_dates = pd.to_datetime(test_data.min_date) 
max_dates = pd.to_datetime(test_data.max_date)

## Here we compute the diff in minutes between first and last purchase
diff_in_time = max_dates - min_dates

## Here we create a dataframe with our results
temp_data = pd.DataFrame(diff_in_time)

## Here we add our results to a column in our data
test_data['diff_first_last'] = temp_data.apply(lambda x: x/np.timedelta64(1, 'm')).fillna(0).astype('int64')

#########################
## STD amount per card ##
#########################

## Here we create a dataframe containing the std from amount spent per card
amt_std = pd.DataFrame(test_data.groupby('cc_num')['amt'].std())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
amt_std['columns'] = amt_std.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
amt_std = amt_std.reset_index().drop(columns = 'columns')

## Here we rename the columns
amt_std.columns = ['cc_num', 'std_amt']

## Here we merge the new data to our table
test_data = amt_std.merge(test_data, on = 'cc_num', how = 'left')

#####################################
## STD amount per card by category ##
#####################################

## Here we create a dataframe containing the std from amount spent per card by category
amt_std = pd.DataFrame(test_data.groupby(['cc_num','category'])['amt'].std())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
amt_std['columns'] = amt_std.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
amt_std = amt_std.reset_index().drop(columns = 'columns')

## Here we rename the columns
amt_std.columns = ['cc_num', 'category','std_amt_by_category']

## Here we merge the new data to our table
test_data = amt_std.merge(test_data, on = ['cc_num','category'], how = 'left')

########################################
## STD distance per card by category ##
#######################################

## Here we create a dataframe containing the std distance per card by category
amt_std = pd.DataFrame(test_data.groupby(['cc_num','category'])['distance'].std())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
amt_std['columns'] = amt_std.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
amt_std = amt_std.reset_index().drop(columns = 'columns')

## Here we rename the columns
amt_std.columns = ['cc_num', 'category','std_dist_by_category']

## Here we merge the new data to our table
test_data = amt_std.merge(test_data, on = ['cc_num','category'], how = 'left')

###########################
## STD distance per card ##
###########################

## Here we create a dataframe containing the std distance per card by category
amt_std = pd.DataFrame(test_data.groupby('cc_num')['distance'].std())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
amt_std['columns'] = amt_std.index

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
amt_std = amt_std.reset_index().drop(columns = 'columns')

## Here we rename the columns
amt_std.columns = ['cc_num', 'std_dist']

## Here we merge the new data to our table
test_data = amt_std.merge(test_data, on = 'cc_num', how = 'left')

################################################
## Purhase amt and dist > avg and > (1,2) std ##
################################################

## Here we create a binary column containing 0 if amount spent is less than avg amount
test_data['purchase_>_avg'] = np.where(test_data['amt'] > test_data['avg_amt'],1,0)

## Here we create a binary column containing 0 if amount spent is less than avg amount per category
test_data['purchase_>_avg_by_category'] = np.where(test_data['amt'] > test_data['avg_by_category'],1,0)

## Here we create a binary column containing 0 if distance is less than avg distance
test_data['purchase_>_distance'] = np.where(test_data['distance'] > test_data['avg_distance'],1,0)

## Here we create a binary column containing 0 if distance is less than avg distance by category
test_data['purchase_>_distance_by_category'] = np.where(test_data['distance'] > test_data['avg_distance_by_category'],1,0)


## Here we classify if the amount spent is higher than 1 std
test_data['amt_>_1_std'] = np.where(test_data.amt > (test_data.avg_amt + test_data.std_amt),1,0)
## Here we classify if the amount spent is higher than 2 std
test_data['amt_>_2_std'] = np.where(test_data.amt > (test_data.avg_amt + 2*test_data.std_amt),1,0)


## Here we classify if the amount spent by category is higher than 1 std
test_data['amt_>_1_std_by_cagegory'] = np.where(test_data.amt > (test_data.avg_by_category + test_data.std_amt_by_category),1,0)
## Here we classify if the amount spent by category is higher than 2 std
test_data['amt_>_2_std_by_cagegory'] = np.where(test_data.amt > (test_data.avg_by_category + 2*test_data.std_amt_by_category),1,0)


## Here we classify if the distance is higher than 1 std
test_data['dist_>_1_std'] = np.where(test_data.distance > (test_data.avg_distance + test_data.std_dist),1,0)
## Here we classify if the distance is higher than 2 std
test_data['dist_>_2_std'] = np.where(test_data.distance > (test_data.avg_distance + 2*test_data.std_dist),1,0)


## Here we classify if the distance by category is higher than 1 std
test_data['dist_>_1_std_by_cagegory'] = np.where(test_data.distance > (test_data.avg_distance_by_category + test_data.std_dist_by_category),1,0)
## Here we classify if the dintance by category is higher than 2 std
test_data['dist_>_2_std_by_cagegory'] = np.where(test_data.distance > (test_data.avg_distance_by_category + 2*test_data.std_dist_by_category),1,0)

#######################
## Uses per category ##
#######################

## Here we create a dataframe containing the uses by category per card
uses_per_category = pd.DataFrame(test_data.groupby('cc_num')['category'].value_counts())

## Here we assign a column containing the index of the dataframe (which is the cc_num)
uses_per_category['columns'] = uses_per_category.index

## Here we rename the columns
uses_per_category.columns = ['uses_per_category','columns']

## Here we reset index to get 0, 1, 2 instead of the cc_nums 
uses_per_category = uses_per_category.reset_index().drop(columns = 'columns')

## Here we merge the new data to our table
test_data = uses_per_category.merge(test_data, on = ['cc_num','category'], how = 'left')


## Here we export the file containing all of the new feature to a csv file. 
train_data.to_csv ("/Users/gabrielmedeiros/Documents/OneDrive/Business analytics/DATA_445_Machine_Learning/CleanTrainData.csv", index = None, header=True)
test_data.to_csv ("/Users/gabrielmedeiros/Documents/OneDrive/Business analytics/DATA_445_Machine_Learning/CleanTestData.csv", index = None, header=True)