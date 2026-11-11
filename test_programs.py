"""Автоматизовані перевірки для практичної роботи №8–9."""
import unittest
from pathlib import Path
import pandas as pd
from task1_dataframe import analyze as analyze_gdp
from task2_bicycle_2015 import analyze as analyze_bikes
from task3_nltk_blake import analyze as analyze_text

ROOT = Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_gdp_58_rows(self):
        df, grouped = analyze_gdp()
        self.assertEqual(len(df), 58)
        self.assertEqual(len(grouped), 2)
        self.assertEqual(df['year'].min(), 1991)
        self.assertEqual(df['year'].max(), 2019)

    def test_bicycles_source_monthly_12(self):
        df = pd.read_csv(ROOT / 'data' / 'montreal_bikes_2015_monthly.csv')
        self.assertEqual(len(df), 12)
        self.assertEqual(len(df.columns) - 1, 18)

    def test_bicycles_total_and_best_month(self):
        total, best = analyze_bikes()
        self.assertEqual(total, 10060290)
        self.assertEqual([item['month'] for item in best], [7, 7, 7])

    def test_text_demo(self):
        result = analyze_text(demo=True)
        self.assertEqual(result['word_count'], 91)
        self.assertEqual(result['unique_words'], 52)
        self.assertEqual(result['top_stemmed'][0], ('pipe', 8))

    def test_reference_blake_summary(self):
        import json
        d = json.loads((ROOT / 'data' / 'blake_summary_reference.json').read_text(encoding='utf-8'))
        self.assertEqual(d['words'], 6817)
        self.assertEqual(d['top_raw'][0], ['the', 439])

if __name__ == '__main__':
    unittest.main(verbosity=2)
