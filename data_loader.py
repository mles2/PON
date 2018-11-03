import csv
import numpy as np

def readcsv(filename,dm):
    with open(filename, 'r') as p:
        my_list = [[int(x) for x in rec] for rec in csv.reader(p, delimiter=dm)]
    return my_list

def load_data(filename):
    data_array = readcsv(filename,',')
    for l in data_array:
        l=np.array(l)

    data_array = np.array(data_array)
    attribute_array = data_array[:,1:-1]
    attribute_array = attribute_array.transpose() / 10

    return data_array, attribute_array