from tests import test_continuous
from src.spaces.bucket_space import BucketSpace


class BucketSpaceTest(test_continuous.TestContinuous):
    bucket = BucketSpace(0.0, 1.0, bucket_width=0.05, epsilon=0.01)

    def test_remove_intervals(self):
        expected = [(0.05, 0.1), (0.15, 0.35), (0.45, 0.6), (0.65, 0.95)]

        self.bucket.remove(0.62)
        self.bucket.remove(1.5)
        self.bucket.remove(0.13)
        self.bucket.remove(0.0)
        self.bucket.remove(1.0)
        self.bucket.remove(0.4)

        assert self.bucket.intervals == expected

    def test_insert_intervals(self):
        expected = [(0.0, 1.0)]

        self.bucket.remove(0.62)
        self.bucket.add(0.63)

        assert self.bucket.intervals == expected

    def test_intervals(self):
        expected = [(0.0, 1.0)]

        print(self.bucket.intervals)

        assert self.bucket.intervals == expected

    def test_bucket_sample(self):
        print(self.bucket.sample())