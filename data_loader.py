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

def convert_array(outputs):
    new = np.zeros((len(outputs),2))
    for x in range(len(outputs)):
        if outputs[x] == 2:
            new[x] = np.array([1,0])
        elif outputs[x] == 4:
            new[x] = np.array([0,1])
    return new

def load_array_from_data(filename):
    data_array = readcsv(filename,',')
    data_array = np.array(data_array)
    inputs = data_array[:,1:-1]
    print(inputs)
    outputs = data_array[:, -1]


    return inputs, convert_array(outputs)