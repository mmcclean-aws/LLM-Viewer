import pandas as pd
import os

def load_and_sort_data(device: str) -> pd.DataFrame:
    """
    Load and sort the device-specific CSV data.
    
    Args:
        device: Name of the device (e.g., 'trainium2')
        
    Returns:
        pd.DataFrame: Sorted dataframe containing the bandwidth data
    """
    # Read the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, f"{device}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found for device {device}")
        
    df = pd.read_csv(csv_path)
    
    # Sort the data by algorithm, type, cores, and size
    df = df.sort_values(by=['algorithm', 'type', 'cores', 'size'])
    
    return df

# Cache the sorted dataframes
_sorted_dfs = {}

def get_algorithm_bandwidth(
    message_size: int,
    algorithm: str,
    core_count: int,
    device: str,
    data_type: str = "bfloat16"
) -> float:
    """
    Lookup algorithm bandwidth from device-specific CSV data.
    Returns bandwidth for largest size entry that is less than message_size.
    If message_size is larger than all entries, returns bandwidth for largest size.
    
    Args:
        message_size: Size of the message in bytes
        algorithm: Algorithm type ('allr', 'allg', 'redsct')
        core_count: Number of cores (4, 8, 16, 32, 64)
        device: Name of the device (e.g., 'trainium2')
        data_type: Data type (default: 'bfloat16')
        
    Returns:
        float: Algorithm bandwidth in GB/s
    """
    global _sorted_dfs
    
    # Load data if not already cached
    if device not in _sorted_dfs:
        _sorted_dfs[device] = load_and_sort_data(device)
    
    df = _sorted_dfs[device]
    
    # Filter by everything except size
    filtered = df[
        (df['algorithm'] == algorithm) &
        (df['cores'] == core_count) &
        (df['type'] == data_type)
    ]
    
    if filtered.empty:
        raise ValueError(f"No matching data found for device={device}, algorithm={algorithm}, core_count={core_count}, data_type={data_type}")
    
    # Get largest size entry less than message_size
    filtered = filtered[filtered['size'] <= message_size]
    if filtered.empty:
        # If message_size is smaller than all entries, use smallest size entry
        filtered = df[
            (df['algorithm'] == algorithm) &
            (df['cores'] == core_count) &
            (df['type'] == data_type)
        ]
    
    # Return bandwidth for largest applicable size and convert to float
    return float(filtered.nlargest(1, 'size')['algbw'].iloc[0])

def get_sorted_data(device: str) -> pd.DataFrame:
    """
    Get the sorted bandwidth data for a specific device.
    
    Args:
        device: Name of the device (e.g., 'trainium2')
        
    Returns:
        pd.DataFrame: Sorted dataframe containing the bandwidth data
    """
    global _sorted_dfs
    if device not in _sorted_dfs:
        _sorted_dfs[device] = load_and_sort_data(device)
    return _sorted_dfs[device]
