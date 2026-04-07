import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from backend.model_training import CLASSES, evaluate_model


class ModelTrainingEvaluationTests(unittest.TestCase):
    def test_evaluate_model_handles_missing_class_in_test_split(self) -> None:
        label_encoder = LabelEncoder()
        label_encoder.fit(CLASSES)

        x_train = np.array(
            [
                [0.0] * 16,
                [0.1] * 16,
                [0.2] * 16,
                [1.0] * 16,
                [1.1] * 16,
                [1.2] * 16,
                [2.0] * 16,
                [2.1] * 16,
                [2.2] * 16,
            ]
        )
        y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        classifier = RandomForestClassifier(n_estimators=20, random_state=42)
        classifier.fit(x_train, y_train)

        x_test = np.array(
            [
                [0.05] * 16,
                [0.15] * 16,
                [1.05] * 16,
                [1.15] * 16,
            ]
        )
        y_test = np.array([0, 0, 1, 1])

        accuracy, matrix, feature_importances = evaluate_model(
            classifier,
            x_test,
            y_test,
            label_encoder,
        )

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(len(feature_importances), 16)


if __name__ == "__main__":
    unittest.main()
