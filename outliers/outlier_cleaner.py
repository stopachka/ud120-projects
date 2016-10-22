#!/usr/bin/python

import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    ### your code goes here

    errors = predictions - net_worths

    threshold = np.percentile(np.absolute(errors), 90)

    return filter(
        lambda (age, net_worth, error): abs(error) <= threshold,
        zip(ages, net_worths, errors)
    )
