"""
This script is used to collect all text files from a given directory and its subdirectories, 
and copy them into a new folder named 'all_transcript' in the same directory.

The script takes one command-line argument:
    1. The path to the directory (str): The path to the directory from which text files are to be collected.

The script uses the os and shutil modules to navigate the directory structure and copy files.

Functions:
    grab_text_files(directory: str) -> None:
        Collects all '.txt' files from the specified directory and its subdirectories,
        and copies them into a new folder named 'all_transcript' in the same directory.

Usage:
    python grab_text_files.py <directory>

Example:
    python grab_text_files.py /path/to/your/directory
"""

import os
import shutil
import sys

def grab_text_files(directory):
    """
    Collects all '.txt' files from the specified directory and its subdirectories,
    and copies them into a new folder named 'all_transcript' in the same directory.

    Args:
        directory (str): The path to the directory from which text files are to be collected.
    """
    # Create a new directory named 'all_transcript'
    new_dir = os.path.join(directory, 'all_transcript')
    os.makedirs(new_dir, exist_ok=True)

    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # Check if the file is a text file
            if filename.endswith('.txt'):
                # Construct full file path
                file_path = os.path.join(dirpath, filename)
                # Copy the file to the new directory
                shutil.copy(file_path, new_dir)

# Check if a command-line argument was provided
if len(sys.argv) != 2:
    print("Usage: python grab_text_files.py <directory>")
else:
    # Call the function with your directory
    grab_text_files(sys.argv[1])