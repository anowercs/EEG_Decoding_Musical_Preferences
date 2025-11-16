import pickle
import numpy as np

# Replace with your actual file path
path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/Figure/eeg_features_full.pkl'

# Load the data from the pickle file
with open(path, 'rb') as f:
    data = pickle.load(f)

# Assuming 'subjects' is stored in data dictionary or as an element
# Adjust this line depending on your data structure
subjects = data['subjects']  # or data.subjects or data['Subjects']

# Convert to numpy array if not already
subjects = np.array(subjects)

# Find unique subjects and their counts
unique_subjects, counts = np.unique(subjects, return_counts=True)

print("All unique subjects:", unique_subjects)
print("Sample counts per subject:", dict(zip(unique_subjects, counts)))

# Print first 50 subjects in data order
print("First 50 subjects in data:", subjects[:50])
