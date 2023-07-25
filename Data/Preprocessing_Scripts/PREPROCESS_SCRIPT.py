import pandas as pd
import re


def clean_cpu_util_data(filename):
    """
    Function to clean the CPU utilization data from a text file and convert it into a DataFrame.

    Parameters:
    filename (str): The name of the file containing CPU utilization data

    Returns:
    pd.DataFrame: DataFrame containing cleaned CPU utilization data
    """
    # Open file and read its contents into 'text_file'
    with open(filename, "r") as f:
        text_file = f.read()

    # Define column names for a DataFrame
    col_names = ['CPU_1_Core_0_Thread_0', 'CPU_1_Core_0_Thread_1', 'CPU_1_Core_1_Thread_0', 'CPU_1_Core_1_Thread_1',
                 'CPU_1_Core_2_Thread_0', 'CPU_1_Core_2_Thread_1',
                 'CPU_1_Core_3_Thread_0', 'CPU_1_Core_3_Thread_1', 'CPU_1_Core_4_Thread_0', 'CPU_1_Core_4_Thread_1',
                 'CPU_1_Core_5_Thread_0', 'CPU_1_Core_5_Thread_1',
                 'CPU_1_Core_6_Thread_0', 'CPU_1_Core_6_Thread_1', 'CPU_1_Core_7_Thread_0', 'CPU_1_Core_7_Thread_1',
                 'CPU_1_Core_8_Thread_0', 'CPU_1_Core_8_Thread_1',
                 'CPU_1_Core_9_Thread_0', 'CPU_1_Core_9_Thread_1', 'CPU_2_Core_0_Thread_0', 'CPU_2_Core_0_Thread_1',
                 'CPU_2_Core_1_Thread_0', 'CPU_2_Core_1_Thread_1',
                 'CPU_2_Core_2_Thread_0', 'CPU_2_Core_2_Thread_1', 'CPU_2_Core_3_Thread_0', 'CPU_2_Core_3_Thread_1',
                 'CPU_2_Core_4_Thread_0', 'CPU_2_Core_4_Thread_1',
                 'CPU_2_Core_5_Thread_0', 'CPU_2_Core_5_Thread_1', 'CPU_2_Core_6_Thread_0', 'CPU_2_Core_6_Thread_1',
                 'CPU_2_Core_7_Thread_0', 'CPU_2_Core_7_Thread_1',
                 'CPU_2_Core_8_Thread_0', 'CPU_2_Core_8_Thread_1', 'CPU_2_Core_9_Thread_0', 'CPU_2_Core_9_Thread_1']

    # Initialize DataFrame with defined column names
    df = pd.DataFrame(columns=col_names)

    # Define the pattern for CPU utilization
    cpu_util_pattern = r"(\d{1,3}\.\d{1,2}) us"

    # initialize the variables used to parse the data correctly
    thread_counter = 0
    outer_index = 0
    core_name = 0
    core_counter = 0
    cpu_num = 1
    has_data = False

    # Parse each line of the file separately
    for line in text_file.split('\n'):
        # For each line, find all CPU utilizations
        for reading in re.findall(cpu_util_pattern, line):
            has_data = True  # Flag to indicate that data is found

            # Add the utilization to the DataFrame
            df.at[outer_index, f'CPU_{cpu_num}_Core_{core_name}_Thread_{thread_counter}'] = reading

            # Since there are only 2 threads, increment the thread counter
            thread_counter += 1

        # We reset the thread count after finding both readings
        thread_counter = 0

        if has_data:
            # Update core name if data has been processed
            core_name += 1

            # Determine which CPU, Core, and timestamp the reading belongs to
            if core_counter < 9:
                cpu_num = 1

            elif core_counter == 9:
                # Reset core name and update CPU number
                core_name = 0
                cpu_num = 2

            # Increment core counter
            core_counter += 1

            # Reset counters if all cores and threads have been processed
            if core_counter == 20:
                cpu_num = 1
                core_name = 0
                core_counter = 0
                outer_index += 1  # Increment index to move to the next row

        # Reset the flag for the next iteration
        has_data = False

    return df


def compute_core_util_averages(df: pd.DataFrame) -> pd.DataFrame:
    # Convert all data to floats to avoid type errors
    df = df.astype(float)

    # Assuming df is your DataFrame
    col_names = ['CPU_1_Core_0_Thread_0', 'CPU_1_Core_0_Thread_1', 'CPU_1_Core_1_Thread_0', 'CPU_1_Core_1_Thread_1',
                 'CPU_1_Core_2_Thread_0', 'CPU_1_Core_2_Thread_1',
                 'CPU_1_Core_3_Thread_0', 'CPU_1_Core_3_Thread_1', 'CPU_1_Core_4_Thread_0', 'CPU_1_Core_4_Thread_1',
                 'CPU_1_Core_5_Thread_0', 'CPU_1_Core_5_Thread_1',
                 'CPU_1_Core_6_Thread_0', 'CPU_1_Core_6_Thread_1', 'CPU_1_Core_7_Thread_0', 'CPU_1_Core_7_Thread_1',
                 'CPU_1_Core_8_Thread_0', 'CPU_1_Core_8_Thread_1',
                 'CPU_1_Core_9_Thread_0', 'CPU_1_Core_9_Thread_1', 'CPU_2_Core_0_Thread_0', 'CPU_2_Core_0_Thread_1',
                 'CPU_2_Core_1_Thread_0', 'CPU_2_Core_1_Thread_1',
                 'CPU_2_Core_2_Thread_0', 'CPU_2_Core_2_Thread_1', 'CPU_2_Core_3_Thread_0', 'CPU_2_Core_3_Thread_1',
                 'CPU_2_Core_4_Thread_0', 'CPU_2_Core_4_Thread_1',
                 'CPU_2_Core_5_Thread_0', 'CPU_2_Core_5_Thread_1', 'CPU_2_Core_6_Thread_0', 'CPU_2_Core_6_Thread_1',
                 'CPU_2_Core_7_Thread_0', 'CPU_2_Core_7_Thread_1',
                 'CPU_2_Core_8_Thread_0', 'CPU_2_Core_8_Thread_1', 'CPU_2_Core_9_Thread_0', 'CPU_2_Core_9_Thread_1']

    # Remove thread information from column names
    group_names = [re.sub("_Thread_[0-9]$", "", col) for col in col_names]

    # Dictionary to store results
    results = {}

    # Iterate through unique groups and calculate average
    for group in set(group_names):
        matching_cols = [col for col, group_name in zip(col_names, group_names) if group_name == group]
        results[group + "_Avg_Util"] = df[matching_cols].mean(axis=1)

    # Convert results to a DataFrame
    df_result = pd.DataFrame(results)

    return df_result


def compute_cpu_util_averages(df: pd.DataFrame) -> pd.DataFrame:
    # Convert all data to floats to avoid type errors
    df = df.astype(float)

    # Assuming df is your DataFrame
    col_names = ['CPU_1_Core_0_Thread_0', 'CPU_1_Core_0_Thread_1', 'CPU_1_Core_1_Thread_0', 'CPU_1_Core_1_Thread_1',
                 'CPU_1_Core_2_Thread_0', 'CPU_1_Core_2_Thread_1',
                 'CPU_1_Core_3_Thread_0', 'CPU_1_Core_3_Thread_1', 'CPU_1_Core_4_Thread_0', 'CPU_1_Core_4_Thread_1',
                 'CPU_1_Core_5_Thread_0', 'CPU_1_Core_5_Thread_1',
                 'CPU_1_Core_6_Thread_0', 'CPU_1_Core_6_Thread_1', 'CPU_1_Core_7_Thread_0', 'CPU_1_Core_7_Thread_1',
                 'CPU_1_Core_8_Thread_0', 'CPU_1_Core_8_Thread_1',
                 'CPU_1_Core_9_Thread_0', 'CPU_1_Core_9_Thread_1', 'CPU_2_Core_0_Thread_0', 'CPU_2_Core_0_Thread_1',
                 'CPU_2_Core_1_Thread_0', 'CPU_2_Core_1_Thread_1',
                 'CPU_2_Core_2_Thread_0', 'CPU_2_Core_2_Thread_1', 'CPU_2_Core_3_Thread_0', 'CPU_2_Core_3_Thread_1',
                 'CPU_2_Core_4_Thread_0', 'CPU_2_Core_4_Thread_1',
                 'CPU_2_Core_5_Thread_0', 'CPU_2_Core_5_Thread_1', 'CPU_2_Core_6_Thread_0', 'CPU_2_Core_6_Thread_1',
                 'CPU_2_Core_7_Thread_0', 'CPU_2_Core_7_Thread_1',
                 'CPU_2_Core_8_Thread_0', 'CPU_2_Core_8_Thread_1', 'CPU_2_Core_9_Thread_0', 'CPU_2_Core_9_Thread_1']

    # Remove core and thread information from column names
    group_names = [re.sub("_Core_[0-9]_Thread_[0-9]$", "", col) for col in col_names]

    # Dictionary to store results
    results = {}

    # Iterate through unique groups and calculate average
    for group in set(group_names):
        matching_cols = [col for col, group_name in zip(col_names, group_names) if group_name == group]
        results[group + "_Avg_Util"] = df[matching_cols].mean(axis=1)

    # Convert results to a DataFrame
    df_result = pd.DataFrame(results)

    return df_result


def clean_cpu_temp_data(filename):
    """
    Function to clean the CPU temperature data from a text file and convert it into a DataFrame.

    Parameters:
    filename (str): The name of the file containing CPU temperature data

    Returns:
    pd.DataFrame: DataFrame containing cleaned CPU temperature data
    """
    # Open file and read its contents into 'text_file'
    with open(filename, "r") as f:
        text_file = f.read()

    # Define column names for a DataFrame
    col_names = ['timestamp', 'CPU_1_Core_0', 'CPU_1_Core_1', 'CPU_1_Core_2', 'CPU_1_Core_3', 'CPU_1_Core_4',
                 'CPU_1_Core_5', 'CPU_1_Core_6', 'CPU_1_Core_7', 'CPU_1_Core_8', 'CPU_1_Core_9', 'CPU_2_Core_0',
                 'CPU_2_Core_1', 'CPU_2_Core_2', 'CPU_2_Core_3', 'CPU_2_Core_4', 'CPU_2_Core_5', 'CPU_2_Core_6',
                 'CPU_2_Core_7', 'CPU_2_Core_8', 'CPU_2_Core_9']

    # Initialize DataFrame with defined column names
    df = pd.DataFrame(columns=col_names)

    # Remove all occurrences of "Â°" from the text file
    text_file = re.sub("Â°", "", text_file)

    # Define the pattern for date and timezone
    date_pattern = r"\D\D\D \D\D\D \d\d \d\d\:\d\d\:\d\d \D\D \D\D\D \d\d\d\d"
    timezone = "PDT "

    # Extract and parse timestamp data
    for idx, temp_reading in enumerate(re.findall(date_pattern, text_file)):
        # Remove timezone and convert the remaining string to datetime
        temp_reading = re.sub(timezone, "", temp_reading)
        dt = pd.to_datetime(temp_reading)
        # Add the datetime to the DataFrame
        df.at[idx, 'timestamp'] = dt

    # Define the pattern for temperature readings
    core_temps_separate_cpu_pattern = r"\D{4} \d{1,2}:\s{7,8}\+\d{2}.\d"
    temp_pattern = r"(\d{2}.\d)"

    # Initialize counting variables necessary for parsing the data to the correct location
    core_counter = 0
    outer_index = 0
    core_name = 0

    # Parse each line of the file separately
    for line in text_file.split('\n'):
        # For each line, find all temperature readings
        for temp_reading in re.findall(core_temps_separate_cpu_pattern, line):
            # Determine which CPU the reading belongs to
            if core_counter < 10:
                cpu_num = 1
            elif core_counter == 10:
                # When moving to CPU 2, reset core_name
                cpu_num = 2
                core_name = 0
            elif core_counter == 20:
                # Reset counters when moving to a new set of readings
                cpu_num = 1
                core_name = 0
                core_counter = 0
                outer_index += 1

            # Extract just the temperature from the reading
            temp = re.search(temp_pattern, temp_reading).group(1)

           # Add the temperature to the DataFrame, constructing the column name based on cpu_num and core_name
            df.at[outer_index, f'CPU_{cpu_num}_Core_{core_name}'] = temp

            # Increment the core_counter and core_name for the next iteration
            core_counter += 1
            core_name += 1

    return df


def compute_cpu_temp_averages(df: pd.DataFrame) -> pd.DataFrame:
    # Save timestamp column to add it back later
    timestamp = df['timestamp']

    # Remove the 'timestamp' column for the conversion
    df_temp = df.drop('timestamp', axis=1)

    # Convert all data to floats to avoid type errors
    df_temp = df_temp.astype(float)

    col_names = ['CPU_1_Core_0', 'CPU_1_Core_1', 'CPU_1_Core_2', 'CPU_1_Core_3', 'CPU_1_Core_4', 'CPU_1_Core_5',
                 'CPU_1_Core_6', 'CPU_1_Core_7', 'CPU_1_Core_8', 'CPU_1_Core_9', 'CPU_2_Core_0', 'CPU_2_Core_1',
                 'CPU_2_Core_2', 'CPU_2_Core_3', 'CPU_2_Core_4', 'CPU_2_Core_5', 'CPU_2_Core_6', 'CPU_2_Core_7',
                 'CPU_2_Core_8', 'CPU_2_Core_9']

    # Remove core information from column names
    group_names = [re.sub("_Core_[0-9]$", "", col) for col in col_names]

    # Dictionary to store results
    results = {}

    # Iterate through unique groups and calculate average
    for group in set(group_names):
        matching_cols = [col for col, group_name in zip(col_names, group_names) if group_name == group]
        results[group+"_Avg_Temp"] = df_temp[matching_cols].mean(axis=1)

    # Convert results to a DataFrame
    df_result = pd.DataFrame(results)

    # Add back the timestamp
    df_result['timestamp'] = timestamp

    return df_result


def clean_gpu_status_data(filename):
    """
    Function to clean the GPU status data from a text file and convert it into a DataFrame.

    Parameters:
    filename (str): The name of the file containing GPU status data

    Returns:
    pd.DataFrame: DataFrame containing cleaned GPU status data
    """
    # Open file and read its contents into 'text_file'
    with open(filename, "r") as f:
        text_file = f.read()

    # Define column names for a DataFrame
    col_names = ["gpu_temp", "gpu_power", "gpu_GRAM", "gpu_util"]

    # Initialize DataFrame with defined column names
    df = pd.DataFrame(columns=col_names)

    # Define the pattern for GPU temperature, power, GRAM, and utilization
    gpu_temp_pattern = r"(\d{1,3})C"
    gpu_power_pattern = r"(\d{1,2})W /"
    gpu_GRAM_pattern = r"(\d{1,4})MiB /"
    gpu_util_pattern = r"(\d{1,3})%"

    # Parse the text file for each GPU parameter

    # Find all GPU temperatures in text file and add to DataFrame
    for idx, reading in enumerate(re.findall(gpu_temp_pattern, text_file)):
        df.at[idx, "gpu_temp"] = reading

    # Find all GPU power readings in text file and add to DataFrame
    for idx, reading in enumerate(re.findall(gpu_power_pattern, text_file)):
        df.at[idx, "gpu_power"] = reading

    # Find all GPU GRAM readings in text file and add to DataFrame
    for idx, reading in enumerate(re.findall(gpu_GRAM_pattern, text_file)):
        df.at[idx, "gpu_GRAM"] = reading

    # Find all GPU utilizations in text file and add to DataFrame
    for idx, reading in enumerate(re.findall(gpu_util_pattern, text_file)):
        df.at[idx, "gpu_util"] = reading

    return df



def clean_shape_of_diagnostic_data(cpu_df, gpu_status_df):
    """
    Function to trim the length of dataframes to match the smallest one.

    Parameters:
    cpu_temp_df (pd.DataFrame): DataFrame with CPU temperature data
    cpu_util_df (pd.DataFrame): DataFrame with CPU utilization data
    gpu_status_df (pd.DataFrame): DataFrame with GPU status data

    Returns:
    Tuple[pd.DataFrame]: Three dataframes with synchronized lengths
    """
    # Get the shapes of all dataframes
    shapes = [cpu_df.shape[0], gpu_status_df.shape[0]]

    # Find the smallest length
    smallest = min(shapes)

    # Return new dataframes of the smallest length
    return cpu_df[:smallest], gpu_status_df[:smallest]


def merge_diagnostic_data(cpu_df, gpu_status_df):
    """
    Function to merge the CPU and GPU dataframes into a single one.

    Parameters:
    cpu_temp_df (pd.DataFrame): DataFrame with CPU temperature data
    cpu_util_df (pd.DataFrame): DataFrame with CPU utilization data
    gpu_status_df (pd.DataFrame): DataFrame with GPU status data

    Returns:
    pd.DataFrame: Merged dataframe of all the input dataframes
    """


    # Join the resultant CPU dataframe with GPU dataframe
    # Set 'timestamp' as the index of the final dataframe
    all_joined_df = cpu_df.join(gpu_status_df).set_index('timestamp')

    return all_joined_df


def main():
    df_thread_util = clean_cpu_util_data("cpu_util.txt")
    df_core_avg_util = compute_core_util_averages(df_thread_util)
    df_cpu_avg_util = compute_cpu_util_averages(df_thread_util)
    df_cpu_core_temp = clean_cpu_temp_data("cpu_temp.txt")
    df_cpu_avg_temp = compute_cpu_temp_averages(df_cpu_core_temp)


    combined_df = df_cpu_avg_temp.join(df_cpu_avg_util)

    # ensure timestamp column is the first column
    combined_df = combined_df[['timestamp'] + [col for col in combined_df.columns if col != 'timestamp']]

    # List of original column names
    temp_cols = ["CPU_1_Avg_Temp", "CPU_2_Avg_Temp"]
    util_cols = ["CPU_1_Avg_Util", "CPU_2_Avg_Util"]

    # New column names
    new_temp_col = "CPU_Avg_Temp"
    new_util_col = "CPU_Avg_Util"

    combined_df[new_temp_col] = combined_df[temp_cols].mean(axis=1)  # Add new column with average temperature
    combined_df[new_util_col] = combined_df[util_cols].mean(axis=1)  # Add new column with average utilization
    combined_df.drop(columns=temp_cols + util_cols, inplace=True)  # Drop original columns

    # Set pandas display option for column width
    pd.set_option('display.max_colwidth', None)
    gpu_status_df = clean_gpu_status_data("gpu_status.txt")

    # Clean shapes of all dataframes
    combined_df, gpu_status_df = clean_shape_of_diagnostic_data(combined_df, gpu_status_df)

    # Merge all the dataframes
    joined_df = merge_diagnostic_data(gpu_status_df, combined_df)

    # get GRAM column in format it will be in on Cloudsim+
    joined_df['gpu_GRAM'] = joined_df['gpu_GRAM'].astype('float32') / 7611 * 100

    # Get filename from user input
    filename = input("Provide a name for the dataset csv file: ")

    # Save the joined dataframe to a csv file
    joined_df.to_csv(f"{filename}.csv", index=False)


main()

