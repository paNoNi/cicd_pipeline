import unittest

import pandas as pd
from sklearn.model_selection import train_test_split

from run import get_dataset, train


class AlgorithmTest(unittest.TestCase):

    def test_data_shape(self):
        json_df = pd.read_json('data/review.json', orient='records')
        data, target = get_dataset(json_df)
        self.assertEqual(len(data), len(target))

    def test_predicted_shape(self):
        json_df = pd.read_json('data/review.json', orient='records')
        data, target = get_dataset(json_df)
        _, _, _, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
        predicted = train(data, target)
        self.assertEqual(len(y_test), len(predicted))
