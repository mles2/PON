import csv

def readcsv(filename,dm):
    with open(filename, 'r') as p:
        my_list = [[int(x) for x in rec] for rec in csv.reader(p, delimiter=',')]
    return my_list