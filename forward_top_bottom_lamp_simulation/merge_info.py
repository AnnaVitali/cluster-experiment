import os
import re

def merge_files(dir1: str, dir2: str, output_dir: str):
        # Print the absolute path of the output directory
    print(f"Creating directory: {os.path.abspath(output_dir)}")

    try:
        # Try to create the output directory
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        # If an error occurs, print it
        print(f"Failed to create directory: {e}")
        return


    # Get all files in the directories
    files1 = {re.search(r'(\d+)', f).group(): f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))}
    files2 = {re.search(r'(\d+)', f).group(): f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))}

    # Iterate over the files in the first directory
    for index in files1.keys():
        # Check if the file with the same index exists in the second directory
        if index in files2:
            # Open the output file
            with open(os.path.join(output_dir, f'case{index}_info.txt'), 'w') as outfile:
                # Open the file from the first directory
                with open(os.path.join(dir1, files1[index]), 'r') as infile:
                    # Write the contents to the output file
                    outfile.write(infile.read())
                # Open the file from the second directory
                with open(os.path.join(dir2, files2[index]), 'r') as infile:
                    # Write the contents to the output file
                    outfile.write(infile.read())

# Call the function with your directories and output directory
merge_files('./dataset/test_info', './dataset/test_optimize_info', 'merged_test')
merge_files('./dataset/validation_info', './dataset/validation_optimize_info', 'merged_validation')
merge_files('./dataset/train_info', './dataset/train_optimize_info', 'merged_train')
