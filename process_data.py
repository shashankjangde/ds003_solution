# this file would conver the txt data into csv data and also adds column names

import csv
import os

import pandas as pd

# below dirs are made assuming you are in the folder in which you have extracted the dataset
parent = "./CMAPSSData/"
data_dir = parent + "processed_data/"
# sorts the data according to train, test and RUL
rul = sorted([i for i in os.listdir(parent) if "RUL_" in i])
train = sorted([i for i in os.listdir(parent) if "train_" in i])
test = sorted([i for i in os.listdir(parent) if "test_" in i])

if data_dir not in os.listdir():
    os.mkdir(data_dir)

# this loop will convert the actualRUL for every corresponding train dataset into a CSV file containing the unit number and the actualRUL
for i in rul:
    header = ["Expected RUL"]
    with open(parent + i, mode="r") as text_file:
        data = [int(j) for j in text_file]

    write_file = data_dir + i.rstrip(".txt") + ".csv"
    with open(write_file, mode="w") as current_file:
        csvwriter = csv.writer(current_file)
        csvwriter.writerow(header)
        csvwriter.writerows([[data[i - 1]] for i in range(1, len(data) + 1)])

# this loop will convert the train and test files to csv files
for i in train + test:
    header = (
        ["Unit", "CurrentRUL"]
        + [f"Operation Setting {j}" for j in range(1, 4)]
        + [f"Sensor {j}" for j in range(1, 22)]
    )
    with open(parent + i, "r") as text_file:
        data = [[float(x) for x in j.split()] for j in text_file]

    write_file = data_dir + i.rstrip(".txt") + ".csv"
    with open(write_file, "w") as current_file:
        csvwriter = csv.writer(current_file)
        csvwriter.writerow(header)
        csvwriter.writerows(data)


# the below loop would create a complete csv file for all scenarios
train_files = [data_dir + i.rstrip(".txt") + '.csv' for i in train]
test_files = [data_dir + i.rstrip(".txt") + '.csv' for i in test]

dataframes = []
for i in train_files:
    df = pd.read_csv(i)
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv(data_dir+"complete_train.csv", index = False)

dataframes = []
for i in test_files:
    df = pd.read_csv(i)
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv(data_dir+"complete_test.csv", index = False)



