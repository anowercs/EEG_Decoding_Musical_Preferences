import pickle
import numpy as np # It's good practice to import numpy as pickled data is often in numpy arrays

# --- Step 1: Specify the path to your .pkl file ---
file_path = '/home/anower/All/Python/Thesis_VS_code/full_epoch/emon/synthetic_data/4.5/eeg_features_full_synthetic.pkl'  # <-- IMPORTANT: Change this to your file's actual path

try:
    # --- Step 2: Load the data from the file in a safe way ---
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ File '{file_path}' loaded successfully!")

    # --- Step 3: Begin inspection ---
    print("\n" + "="*40)
    print("INSPECTING THE LOADED OBJECT")
    print("="*40)

    # A) Find out the overall type of the object
    print(f"\nThe loaded object is a: {type(data)}")

    # B) If it's a dictionary (the most likely case)
    if isinstance(data, dict):
        print("\nIt's a DICTIONARY. Here are its keys:")
        print(f"--> {list(data.keys())}")

        # Now, let's inspect the contents of each key
        for key, value in data.items():
            print(f"\n--- Details for key: '{key}' ---")
            print(f"  - The value is a: {type(value)}")
            if isinstance(value, np.ndarray):
                print(f"  - It's a NumPy array with shape: {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"  - It's a list/tuple with length: {len(value)}")

    # C) If it's a list or tuple
    elif isinstance(data, (list, tuple)):
        print(f"\nIt's a LIST/TUPLE with {len(data)} elements.")
        if len(data) > 0:
            print(f"The first element's type is: {type(data[0])}")

    # D) If it's just a single NumPy array
    elif isinstance(data, np.ndarray):
        print(f"\nIt's a NUMPY ARRAY with shape: {data.shape}")

except FileNotFoundError:
    print(f"❌ ERROR: The file '{file_path}' was not found. Please check the path.")
except Exception as e:
    print(f"❌ An error occurred while loading or inspecting the file: {e}")