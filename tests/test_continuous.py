from unittest import TestCase


class TestContinuous(TestCase):

    def test_imports(self):
        from src.spaces.bucket_space import BucketSpace
        from src.spaces.interval_union import IntervalUnionTree
