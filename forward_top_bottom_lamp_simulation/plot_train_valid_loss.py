import matplotlib.pyplot as plt
import re

# Initialize lists to store the data
epochs = []
training_losses = []
validation_losses = []

# Open the log file and read line by line
with open('./outputs/launch.log', 'r') as file:
    for line in file:
        # Use regex to find lines with the loss information
        match_train = re.search(r'epoch:\s*(\d+),\s*loss:\s*(\d+\.\d+e[+-]\d+)', line)
        if match_train:
            epochs.append(int(match_train.group(1)))
            training_losses.append(float(match_train.group(2)))

        match_val = re.search(r'Validation\s+loss:\s*(\d+\.\d+e[+-]\d+)', line)
        if match_val:
            validation_losses.append(float(match_val.group(1)))

# Check if losses were found, and print a message if not
if not training_losses:
    print("Warning: No training losses found in the log file. Check your regex pattern.")
if not validation_losses:
    print("Warning: No validation losses found in the log file. Check your regex pattern.")

# Plot the training and validation loss (if available)
if training_losses and validation_losses:
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
