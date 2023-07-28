import os
import subprocess

# Get the current working directory
cwd = os.getcwd()

# Specify the relative path to the Python script
relative_path = 'GB_Automate_restaurant.py'

# Create the full path by joining the current working directory and the relative path
full_path = os.path.join(cwd, relative_path)

# Build the command to execute
command = f'python {full_path}'

try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
