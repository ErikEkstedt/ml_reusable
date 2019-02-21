import json
import csv


<<<<<<< HEAD
def read_csv(path, delimiter=','):
=======
def read_csv(path, delimiter):
>>>>>>> a8a767dc28fe0aab1c2f0328c787c32fab92801f
    data = []
    with open(path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        for row in csv_reader:
            data.append(row)
    return data


def read_json(path):
    with open(path, 'r') as f:
        data = json.loads(f.read())
    return data


def write_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        data = json.dump(data, jsonfile, ensure_ascii=False)
    return data
