import json
import csv


def write_txt(txt, name):
    """
    Argument:
        txt:    list of strings
        name:   filename
    """
    with open(name, "w") as f:
        f.write("\n".join(txt))


def read_csv(path, delimiter=","):
    data = []
    with open(path, "r") as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        for row in csv_reader:
            data.append(row)
    return data


def read_json(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.loads(f.read())
    return data


def write_json(data, filename):
    with open(filename, "w", encoding="utf-8") as jsonfile:
        data = json.dump(data, jsonfile, ensure_ascii=False)
    return data
