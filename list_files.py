import os
import csv
import sys
from datetime import datetime

def list_files_to_csv(directory, csv_file):
    """
    This function takes a directory and a csv file as input. It walks through the directory and its subdirectories,
    collects information about each file, and writes this information to the csv file.

    Parameters:
    directory (str): The directory to search.
    csv_file (str): The csv file to write to.
    """

    # Define the column names for the csv file
    columns = ['File Name', 'File Path', 'File Size (Bytes)', 'Last Modified']

    # Open the csv file in write mode
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the column names to the csv file
        writer.writerow(columns)

        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            # For each file in the directory
            for name in files:
                # Get the full path of the file
                file_path = os.path.join(root, name)
                # Get the size of the file
                file_size = os.path.getsize(file_path)
                # Get the last modified time of the file
                last_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                # Write the file information to the csv file
                writer.writerow([name, file_path, file_size, last_modified])

if __name__ == "__main__":
    # Check if the correct number of command line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python aclu_file_list.py [input_path] [output_path]")
        sys.exit(1)

    # Get the input directory and output csv file from the command line arguments
    input_directory = sys.argv[1]
    output_csv = sys.argv[2]

    # Call the function to list the files to the csv file
    list_files_to_csv(input_directory, output_csv)
    print(f"File list successfully written to {output_csv}")
