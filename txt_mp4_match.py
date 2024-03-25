import os
import csv
import sys

def match_files(directory):
    """
    This function takes a directory path as input and finds all .txt and .mp4 files in that directory.
    It then writes these file names into a CSV file in the same directory. The CSV file has two columns,
    one for .txt files and one for .mp4 files. If a file doesn't have a match, the corresponding cell in
    the other column is left blank.
    """
    # Get all files in the directory
    files = os.listdir(directory)

    # Separate txt and mp4 files
    txt_files = sorted([f for f in files if f.endswith('.txt')])
    mp4_files = sorted([f for f in files if f.endswith('.mp4')])

    # Prepare data for CSV
    data = [['TXT Files', 'MP4 Files']]
    for txt, mp4 in zip(txt_files, mp4_files):
        data.append([txt, mp4])

    # If there are more txt files
    for txt in txt_files[len(mp4_files):]:
        data.append([txt, ''])

    # If there are more mp4 files
    for mp4 in mp4_files[len(txt_files):]:
        data.append(['', mp4])

    # Write data to CSV
    with open(os.path.join(directory, 'output.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"CSV file has been written to {os.path.join(directory, 'output.csv')}")

# Check if directory path is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <directory_path>")
else:
    # Call the function with the directory path
    match_files(sys.argv[1])
