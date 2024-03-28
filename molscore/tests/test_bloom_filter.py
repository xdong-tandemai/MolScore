import os
import json
import unittest
from importlib import resources

from molscore.tests.base_tests import BaseTests
from molscore.tests.mock_generator import MockGenerator
from molscore.scoring_functions.bloom_filter import BloomFilter

class TestBloomFilter(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):

        # Instantiate
        smiles_path = resources.files('molscore.data') / 'sample.smi'
        bloom_path = resources.files('molscore.data') / 'sample.bloom'
        cls.obj = BloomFilter
        cls.inst = BloomFilter(
            prefix='test',
            smiles_path=None,
            bloom_path=bloom_path,
            canonize=True,
            fpr=0.01,
            n_jobs=4,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"\nBloomFilter Output:\n{json.dumps(cls.output, indent=2)}\n")