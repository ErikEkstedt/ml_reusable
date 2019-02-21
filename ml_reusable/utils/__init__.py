import torch
import json 
import csv 
from datetime import datetime
from os import makedirs
from os.path import join


def read_csv(path, delimiter=','):
    data = []
    with open(path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        for row in csv_reader:
            data.append(row)
    return data


def read_json(path):
    with open(path, 'r') as f:
        dialogue = json.loads(f.read())
    return dialogue


def write_json(dialogue, filename):
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        dialogue = json.dump(dialogue, jsonfile, ensure_ascii=False)
    return dialogue
