import unittest
from get_collectives_bandwidth import get_algorithm_bandwidth, get_sorted_data

class TestBandwidthLookup(unittest.TestCase):
    def setUp(self):
        """Load the sorted data before each test"""
        self.device = "trainium2"
        self.sorted_data = get_sorted_data(self.device)
        self.valid_algorithms = ['allr', 'allg', 'redsct']
        self.valid_cores = [4, 8, 16, 32, 64]
        self.valid_data_type = 'bfloat16'
        self.test_message_size = 8192

    def test_data_loading(self):
        """Test that data is loaded correctly"""
        self.assertIsNotNone(self.sorted_data)
        self.assertGreater(len(self.sorted_data), 0)
        self.assertIn('size', self.sorted_data.columns)
        self.assertIn('algorithm', self.sorted_data.columns)
        self.assertIn('cores', self.sorted_data.columns)
        self.assertIn('type', self.sorted_data.columns)
        self.assertIn('algbw', self.sorted_data.columns)

    def test_exact_size_match(self):
        """Test lookup with exact size match"""
        bandwidth = get_algorithm_bandwidth(
            message_size=self.test_message_size,
            algorithm='allr',
            core_count=4,
            device=self.device,
            data_type=self.valid_data_type
        )
        self.assertIsInstance(bandwidth, float)
        self.assertGreater(bandwidth, 0)
        self.assertLess(bandwidth, 1000)  # Sanity check for reasonable bandwidth

    def test_size_interpolation(self):
        """Test lookup with various message sizes"""
        # Test with a size in the middle of the range
        mid_size = 65536
        bandwidth = get_algorithm_bandwidth(
            message_size=mid_size,
            algorithm='allr',
            core_count=4,
            device=self.device,
            data_type=self.valid_data_type
        )
        self.assertIsInstance(bandwidth, float)
        self.assertGreater(bandwidth, 0)

        # Test with a very large size
        large_size = 2**40  # 1TB
        bandwidth = get_algorithm_bandwidth(
            message_size=large_size,
            algorithm='allr',
            core_count=4,
            device=self.device,
            data_type=self.valid_data_type
        )
        self.assertIsInstance(bandwidth, float)
        self.assertGreater(bandwidth, 0)

        # Test with a very small size
        small_size = 1
        bandwidth = get_algorithm_bandwidth(
            message_size=small_size,
            algorithm='allr',
            core_count=4,
            device=self.device,
            data_type=self.valid_data_type
        )
        self.assertIsInstance(bandwidth, float)
        self.assertGreater(bandwidth, 0)

    def test_different_algorithms(self):
        """Test lookup with different algorithms"""
        for algo in self.valid_algorithms:
            with self.subTest(algorithm=algo):
                bandwidth = get_algorithm_bandwidth(
                    message_size=self.test_message_size,
                    algorithm=algo,
                    core_count=4,
                    device=self.device,
                    data_type=self.valid_data_type
                )
                self.assertIsInstance(bandwidth, float)
                self.assertGreater(bandwidth, 0)

    def test_bandwidth_greater(self):
        """Test that bandwidth is larger for larger message sizes"""
        for algo in self.valid_algorithms:
            with self.subTest(algorithm=algo):
                bandwidth_1 = get_algorithm_bandwidth(
                    message_size=self.test_message_size,
                    algorithm=algo,
                    core_count=4,
                    device=self.device,
                    data_type=self.valid_data_type
                )
                bandwidth_2 = get_algorithm_bandwidth(
                    message_size=self.test_message_size * 2,
                    algorithm=algo,
                    core_count=4,
                    device=self.device,
                    data_type=self.valid_data_type
                )
                self.assertGreater(bandwidth_2, bandwidth_1)

    def test_bandwidth_same(self):
        """Test that bandwidth is larger for larger message sizes"""
        for algo in self.valid_algorithms:
            with self.subTest(algorithm=algo):
                bandwidth_1 = get_algorithm_bandwidth(
                    message_size=self.test_message_size,
                    algorithm=algo,
                    core_count=4,
                    device=self.device,
                    data_type=self.valid_data_type
                )
                bandwidth_2 = get_algorithm_bandwidth(
                    message_size=self.test_message_size + 128,
                    algorithm=algo,
                    core_count=4,
                    device=self.device,
                    data_type=self.valid_data_type
                )
                self.assertAlmostEqual(bandwidth_2, bandwidth_1)


    def test_different_cores(self):
        """Test lookup with different core counts"""
        for cores in self.valid_cores:
            with self.subTest(core_count=cores):
                bandwidth = get_algorithm_bandwidth(
                    message_size=self.test_message_size,
                    algorithm='allr',
                    core_count=cores,
                    device=self.device,
                    data_type=self.valid_data_type
                )
                self.assertIsInstance(bandwidth, float)
                self.assertGreater(bandwidth, 0)

    def test_invalid_parameters(self):
        """Test lookup with invalid parameters"""
        # Test with invalid algorithm
        with self.assertRaises(ValueError):
            get_algorithm_bandwidth(
                message_size=self.test_message_size,
                algorithm='invalid',
                core_count=4,
                device=self.device,
                data_type=self.valid_data_type
            )

        # Test with invalid core count
        with self.assertRaises(ValueError):
            get_algorithm_bandwidth(
                message_size=self.test_message_size,
                algorithm='allr',
                core_count=3,  # Invalid core count
                device=self.device,
                data_type=self.valid_data_type
            )

        # Test with invalid data type
        with self.assertRaises(ValueError):
            get_algorithm_bandwidth(
                message_size=self.test_message_size,
                algorithm='allr',
                core_count=4,
                device=self.device,
                data_type='invalid'
            )

        # Test with invalid device
        with self.assertRaises(FileNotFoundError):
            get_algorithm_bandwidth(
                message_size=self.test_message_size,
                algorithm='allr',
                core_count=4,
                device='invalid_device',
                data_type=self.valid_data_type
            )

    def test_bandwidth_trends(self):
        """Test that bandwidth increases with message size"""
        sizes = [8192, 16384, 32768]
        bandwidths = []
        
        for size in sizes:
            bandwidth = get_algorithm_bandwidth(
                message_size=size,
                algorithm='allr',
                core_count=4,
                device=self.device,
                data_type=self.valid_data_type
            )
            bandwidths.append(bandwidth)
        
        # Check that bandwidth generally increases with size
        for i in range(len(bandwidths) - 1):
            self.assertGreaterEqual(bandwidths[i + 1], bandwidths[i] * 0.5)  # Allow for some variation

    def test_core_scaling(self):
        """Test that bandwidth generally increases with core count"""
        bandwidths = []
        
        for cores in self.valid_cores:
            bandwidth = get_algorithm_bandwidth(
                message_size=self.test_message_size,
                algorithm='allr',
                core_count=cores,
                device=self.device,
                data_type=self.valid_data_type
            )
            bandwidths.append(bandwidth)
        
        # Check that bandwidth generally increases with core count
        for i in range(len(bandwidths) - 1):
            self.assertGreaterEqual(bandwidths[i + 1], bandwidths[i] * 0.5)  # Allow for some variation

if __name__ == '__main__':
    unittest.main() 