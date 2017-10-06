import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# *********************** first step is data cleaning ***********************
def data_cleanining(df_new):

    # 1. remove  since almost they are Null(missing values)
    del df_new["Ehail_fee"]

    # 2. replace missing value in "Trip_type" by most frequent value which is 1
    mfv_t = df_new['Trip_type '].value_counts().idxmax()
    df_new['Trip_type '].replace(np.NaN, 1,inplace=True)

    # 3.replace invalid value in "RateCodeID" which is 99 by most frequent value which is 1
    # Rate Code values include: 1, 2, 3, 4, 5, 6
    mfv_r = df_new['RateCodeID'].value_counts().idxmax()
    df_new.RateCodeID[~((df_new.RateCodeID>=1) & (df_new.RateCodeID<=6))] = mfv_r

    # 4. replace invalid values in "Extra" by most frequent value which is 0
    # Extra can have only these values: 0, 0.50 and 1
    mfv_e = df_new['Extra'].value_counts().idxmax()
    df_new.Extra[~((df_new.Extra==0.5) |(df_new.Extra==1) | (df_new.RateCodeID==0.5))] = mfv_e

    # 5. replace invalid values in "Passenger_count" which is 0 by most frequent value which is 1
    mfv_p = df_new['Passenger_count'].value_counts().idxmax()
    df_new.Passenger_count[~((df_new.Passenger_count> 0))] = mfv_p

    # 6. replace values in "Trip_distance" <=0 by median
    med_td = df_new['Trip_distance'].median()
    df_new.Trip_distance[(df_new.Trip_distance <= 0)] = med_td

    # 7. Negative values found and replaced by their abs
    df_new.Fare_amount = df_new.Fare_amount.abs()
    df_new.Tip_amount = df_new.Tip_amount.abs()
    df_new.Total_amount = df_new.Total_amount.abs()
    df_new.Tolls_amount = df_new.Tolls_amount.abs()
    df_new.improvement_surcharge = df_new.improvement_surcharge.abs()
    df_new.MTA_tax = df_new.MTA_tax.abs()

    # 8. replace values in "Fare_amount" <  2.5 by median
    med_fm= df_new['Fare_amount'].median()
    df_new.Fare_amount[(df_new.Fare_amount < 2.5)] = med_fm

    # 9. replace values in "Total_amount" <  2.5 by median
    med_tm= df_new['Total_amount'].median()
    df_new.Total_amount[(df_new.Total_amount < 2.5)] = med_tm

    # 10. Assuming 0 = N and 1 = Y, change N to 0 and Y to 1
    df_new['Store_and_fwd_flag'].replace('Y', 1, inplace=True)
    df_new['Store_and_fwd_flag'].replace('N', 0, inplace=True)

    # 11. converting the "lpep_pickup_datetime" and "Lpep_dropoff_datetime" to DateTime series
    df_new.loc[:, "lpep_pickup_datetime"] = pd.to_datetime(df_new.loc[:, "lpep_pickup_datetime"])
    df_new.loc[:, "Lpep_dropoff_datetime"] = pd.to_datetime(df_new.loc[:, "Lpep_dropoff_datetime"])

    # 12. remove those rows that "payment_type" is not with Credit Card (1)
    df_new = df_new[df_new.Payment_type == 1]
    return df_new

# *********************** Second step is feature engineering ***********************
def data_engin(df_new):

    # 1. creating time varibales from "lpep_pickup_datetime" and "Lpep_dropoff_datetime"
    # int [0-23], hour the day a transaction was done
    df_new.loc[:, "hour_p"] = df_new.loc[:, "lpep_pickup_datetime"].dt.hour
    df_new.loc[:, "hour_d"] = df_new.loc[:, "Lpep_dropoff_datetime"].dt.hour
    # int [0-6], day of the week a transaction was done
    df_new.loc[:, "weekday_p"] = df_new.loc[:, "lpep_pickup_datetime"].dt.weekday
    df_new.loc[:, "weekday_d"] = df_new.loc[:, "Lpep_dropoff_datetime"].dt.weekday
    #Month_day: int [0-30], day of the month a transaction was done
    df_new.loc[:, "monthday_p"] = df_new.loc[:, "lpep_pickup_datetime"].dt.day
    df_new.loc[:, "monthday_d"] = df_new.loc[:, "Lpep_dropoff_datetime"].dt.day
    # Trip Duration in seconds
    df_new.loc[:,'trip_duration'] = df_new.loc[:, "Lpep_dropoff_datetime"] - df_new.loc[:, "lpep_pickup_datetime"]
    df_new.loc[:,'trip_duration'] = df_new.loc[:,'trip_duration'].dt.total_seconds()/60
    # removing "lpep_pickup_datetime" and "Lpep_dropoff_datetime"
    del df_new["lpep_pickup_datetime"]
    del df_new["Lpep_dropoff_datetime"]


    # 2. speed variable
    df_new.loc[:,'speed_mph'] = df_new.Trip_distance/(df_new.trip_duration/60)
    # remove transactions that speed in not in valid range
    df_new = df_new[((df_new.speed_mph>0) & (df_new.speed_mph<=240))]
    # 3. location variable

    # 4.  create tip percentage variable
    df_new['tip_percentage'] = 100*df_new.Tip_amount/df_new.Fare_amount
    return df_new
def find_optimized_number_est(data_sample):
    """
        this function finds the optimized number of estimators of the model
        data_sample: pandas data frame
        return integer as optimized # of estimator
    """
    data = data_sample.ix[:, data_sample.columns != 'tip_percentage']
    target = data_sample['tip_percentage']
    # randomly hold out 20% of the data as test set
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, target, test_size=.20, random_state=0)
    mse_final = []
    for j in range(50, 220, 20):
        rf = RandomForestRegressor(n_estimators=j)
        rf.fit(X_train, y_train)
        predicted = rf.predict(X_test)
        mse = mean_squared_error(y_test, predicted)
        mse_final.append((j, mse))
    min_mse = min(mse_final, key = lambda t: t[1])
    print(min_mse[0])
    return min_mse[0]
def evaluate_prediction_with_RFR(df_sample, n_opt, k):

    """
        Build and Evaluate the model using k-fold cross-validation
        param data: pandas dataframe
        param n_opt: optimised number of estimator
        param k: k_fold CV
        return Mean Squared error with 3 fold cross_validation

    """
    data = df_sample.ix[:, df_sample.columns != 'tip_percentage']
    target = df_sample['tip_percentage']
    mse_avg = []
    for i in range(k): # repeat the procedure k times to get more precise results

        # for each iteration, randomly hold out 20% of the data as test set
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data, target, test_size=.20, random_state=0)
        rf = RandomForestRegressor(n_estimators= n_opt)
        rf.fit(X_train, y_train)
    #         print(rf.feature_importances_)
        predicted = rf.predict(X_test)
        print("score", rf.score(X_test, y_test))
        mse = mean_squared_error(y_test, predicted)
        print("Mean Squared error", mse)
        mse_avg.append(mse)
    return np.mean(mse_avg)
def save_optimized_prediction_model(df_sample, n_opt):
    """
    build the prediction model with all the samples and save
    the model as pickle
    """
    data = df_sample.ix[:, df_sample.columns != 'tip_percentage']
    target = df_sample['tip_percentage']
    rf = RandomForestRegressor(n_estimators= n_opt)
    rf.fit(data, target)
    # save the model to disk
    filename = 'pre_tip.pkl'
    pickle.dump(rf, open(filename, 'wb'))
def predict_tip(test):
    """
        this function loads the model from disk and predict the tip for test data
        param test: pandas data frame that is cleaned and feature engineering is done
    """
    # load the model from disk
    filename = 'pre_tip.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    X_test = test.ix[:, test.columns != 'tip_percentage']
    Y_test = test['tip_percentage']
    predicted = loaded_model.predict(X_test)
    print("score", loaded_model.score(X_test, Y_test ))
    mse = mean_squared_error(Y_test , predicted)
    print("Mean Squared error", mse)
    return predicted
def predict(test):
    """
        test data is pandas dataframe and does not have tip_percentage
        predicted result are saved into a csv file which is named result.csv
    """
    print("clean data")
    test = data_cleanining(test)
    print("add features")
    test  = data_engin(test)
    print("predict tips")
    att = ["Fare_amount", "Total_amount", "Tip_amount", "trip_duration", "speed_mph", "tip_percentage"]
    predicted = predict_tip(test[att])
    pre = pd.DataFrame(predicted,columns=['predicted'])
    print("save results into csv file")
    pre.to_csv('result.csv',index=True)
if __name__ == '__main__':
    df = pd.read_csv("green_tripdata_2015-09.csv")
    test =df.loc[np.random.choice(df.index,size=100000,replace=False)]
    predict(test)
