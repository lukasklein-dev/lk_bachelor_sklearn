import unittest
import xml.etree.ElementTree as ET
import sys
import os

"""
Enter in the terminal to run and test the script:
PYTHONPATH=. python -m unittest scripts.tests.fm_test
"""

# Projekt-Root zum sys.path hinzufügen, damit "scripts" importiert werden kann
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import scripts.xml_fm_parser as xml_fm_parser

class TestFeatureModelTransformer(unittest.TestCase):
    def test_simple_and_group(self):
        xml = """
        <struct>
            <and name="root">
                <feature name="A"/>
                <feature name="B"/>
            </and>
        </struct>
        """
        root = ET.fromstring(xml)
        features = xml_fm_parser.parse_feature_model(root[0])
        names = {f['name'] for f in features}
        self.assertIn('A', names)
        self.assertIn('B', names)
        self.assertTrue(all(f['parent'] == 'root' for f in features))

if __name__ == "__main__":
    unittest.main()