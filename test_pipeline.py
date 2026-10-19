import sys, tempfile, unittest, warnings
from pathlib import Path
import pandas as pd
import numpy as np
from feature_utils import validate_input_frame, prepare_features, REQUIRED_INPUT_COLUMNS, TARGET
from predict import predict

ROOT = Path(__file__).parent
class TestRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = pd.read_csv(ROOT / "sample_input.csv")

    def test_normal_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            out = predict(ROOT / "sample_input.csv", Path(td) / "out.csv")
            self.assertEqual(len(out), len(self.sample))
            self.assertTrue(np.isfinite(out[TARGET].to_numpy()).all())
            self.assertTrue((Path(td) / "out.csv").is_file())

    def test_missing_required_column_rejected(self):
        bad = self.sample.drop(columns=["Load_Type"])
        with self.assertRaisesRegex(ValueError, "Missing required column"):
            validate_input_frame(bad)

    def test_invalid_timestamp_rejected(self):
        bad = self.sample.copy()
        bad.loc[0, "date"] = "NOT_A_DATE"
        with self.assertRaisesRegex(ValueError, "Invalid timestamp"):
            validate_input_frame(bad)

    def test_empty_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "no data rows"):
            validate_input_frame(pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS))

    def test_duplicate_timestamp_rejected(self):
        bad = pd.concat([self.sample.iloc[[0]], self.sample.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate timestamps"):
            validate_input_frame(bad)

    def test_inf_numeric_is_cleaned_and_warned(self):
        bad = self.sample.copy()
        bad.loc[0, "CO2(tCO2)"] = np.inf
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clean = validate_input_frame(bad)
        self.assertTrue(pd.isna(clean.loc[0, "CO2(tCO2)"]))
        self.assertTrue(len(caught) > 0)

    def test_invalid_numeric_text_is_warned(self):
        bad = self.sample.copy()
        bad.loc[0, "NSM"] = "not-number"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clean = validate_input_frame(bad)
        self.assertTrue(clean["NSM"].isna().any())
        self.assertTrue(len(caught) > 0)

    def test_target_not_required_for_prediction(self):
        # The supplied sample input intentionally has no target.
        self.assertNotIn(TARGET, self.sample.columns)
        self.assertEqual(len(self.sample), len(predict(ROOT / "sample_input.csv", ROOT / "test_output.csv")))
        (ROOT / "test_output.csv").unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main(verbosity=2)
