#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

def data_points():
    return len(enron_data)

def num_features():
    return len(enron_data.itervalues().next())

def num_poi():
    return len(filter(lambda p : p['poi'] == 1, enron_data.itervalues()))

def stock_by_james():
    return enron_data['PRENTICE JAMES']['total_stock_value']

def colwell_messages():
    return enron_data['COLWELL WESLEY']['from_this_person_to_poi']

def skilling_stock_options():
    return enron_data['SKILLING JEFFREY K']['exercised_stock_options']

def most_money():
    def payments(k):
        return [k, enron_data[k]['total_payments']]

    return [
        payments('SKILLING JEFFREY K'),
        payments('LAY KENNETH L'),
        payments('FASTOW ANDREW S')
    ]

def len_with(f):
    return len(
        filter(
            f,
            enron_data.itervalues()
        )
    )

def percent_nan():
    nan_len = len_with(lambda p : p['total_payments'] == 'NaN')
    total = len(enron_data.itervalues())
    return nan_len / total

def percent_poi_nan():
    vs = enron_data.itervalues()
    pois = filter(lambda p : p['poi'] == 1, vs)
    pois_nan = filter(lambda p : p['total_payments'] == 'NaN', pois)

    return len(pois_nan) / len(pois)
