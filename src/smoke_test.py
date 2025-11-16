# src/smoke_test.py
import sys
print("python:", sys.version)
failed = []
for pkg in ("h5py", "numpy", "matplotlib", "scipy", "seaborn", "pandas", "tqdm", "sklearn"):
    try:
        __import__(pkg)
    except Exception as e:
        failed.append((pkg, str(e)))
if not failed:
    print("All core imports OK")
else:
    print("Import problems for:", failed)

