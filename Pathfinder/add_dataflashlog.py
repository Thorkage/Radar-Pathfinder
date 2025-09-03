from helper_functions import *

def find_dataflashlog(UWIBASS):
    """
    
    """
    
    uwibasstime = UWIBASS.datetime_timestamp
    log_paths = [os.path.join(UWIBASS.data_path, p) for p in UWIBASS.config_paths['dataflashlogs']]
    
    for log_path in log_paths:
        log_files = [os.path.join(log_path, f) for f in os.listdir(log_path) if f.endswith('_processed.csv') and not f.startswith('._')]
        log_files = sorted(log_files)

        for file in log_files:
            file_path = os.path.join(log_path, file)
            # print(f"Checking file: {file_path}")
            first_row = pd.read_csv(file_path, nrows=1, parse_dates=['timestamp'])

            # Read the last line of the CSV file (excluding header)
            with open(file_path, 'rb') as f:
                f.seek(-1024, 2)  # Go to near the end of file (adjust 1024 if needed)
                lines = f.readlines()
                last_line = lines[-1].decode()
            with open(file_path, 'r') as f:
                header = f.readline().strip().split(',')
                
            last_row = pd.read_csv(io.StringIO('\n'.join([','.join(header), last_line])), parse_dates=['timestamp'])

            start_time = first_row['timestamp'].iloc[0]
            end_time = last_row['timestamp'].iloc[0]
            
            if uwibasstime[0] >= start_time and uwibasstime[-1] <= end_time:
                UWIBASS.dataflashlog_path = file_path
                return UWIBASS, file
    print(f"Dataflashlog not found for {UWIBASS.dataset_name} in any of the provided paths.")
    
    return None, None
            
            
            
def load_dataflashlog(uwibass):
    """
    Load the processed dataflashlog for the given UWiBaSS object.
    """
    df = pd.read_csv(uwibass.dataflashlog_path,
                     parse_dates=['timestamp'])
    return df



def attach_dataflashlog(uwibass, df_dataflashlog, columns):
    """
    For each datetime_timestamp in uwibass, find the nearest timestamp in df_dataflashlog,
    and extract the specified columns. The extracted columns are added to the uwibass object.

    Args:
        uwibass: An object with a 'datetime_timestamp' attribute (array-like of datetimes).
        df_dataflashlog: DataFrame with a 'timestamp' column and data columns.
        columns: List of column names (strings) to extract from df_dataflashlog.

    Returns:
        None. The specified columns are added as attributes to uwibass.
    """
    
    
    
    # Find unique uwibass.df_dataflashlog values not already in df_dataflash['timestamp']
    # existing_timestamps = set(df_dataflashlog['timestamp'])
    # uwibass_unique = [dt for dt in uwibass.datetime_timestamp if dt not in existing_timestamps]

    # Create a DataFrame with these new timestamps, fill other columns with NaN

    df_extra = pd.DataFrame({'timestamp':  uwibass.datetime_timestamp})
    # print(len(df_extra))
    # Add all columns from df_dataflash except 'timestamp', filled with NaN
    for col in df_dataflashlog.columns:
        if col != 'timestamp':
            df_extra[col] = np.nan
            
    # Concatenate and sort by timestamp
    df_dataflashlog['source'] = 'dataflash'
    df_extra['source'] = 'uwibass'

    merged = pd.concat([df_dataflashlog, df_extra], ignore_index=True)
    merged = merged.sort_values('timestamp').reset_index(drop=True)

    merged.set_index('timestamp', inplace=True)
    merged = merged.interpolate(method='time')
    merged.reset_index(inplace=True)

    merged = merged.loc[
                        # (merged['timestamp'] >= df_extra['timestamp'].min()) &\
                        # (merged['timestamp'] <= df_extra['timestamp'].max()) &\
                        (merged['source'] == 'uwibass')
                        ]

    for col in columns:
        col_name = col.replace('.', '_')
        setattr(uwibass, col_name, merged[col].values)
        
    return uwibass



    
    
    