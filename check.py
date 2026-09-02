import pandas as pd

print("Checking secom.data...")
try:
    X = pd.read_csv('secom.data', sep=' ', header=None, nrows=5)
    print(f"✅ Loaded — shape: {X.shape}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\nChecking secom_labels.data...")
try:
    L = pd.read_csv('secom_labels.data', sep=' ', header=None,
                    names=['label','timestamp'], nrows=5)
    print(f"✅ Loaded — shape: {L.shape}")
    print(L.head())
except Exception as e:
    print(f"❌ Error: {e}")

print("\nFull shape check...")
X_full = pd.read_csv('secom.data', sep=' ', header=None)
L_full = pd.read_csv('secom_labels.data', sep=' ', header=None,
                     names=['label','timestamp'])
print(f"Features : {X_full.shape} — expected (1567, 591)")
print(f"Labels   : {L_full.shape} — expected (1567, 2)")
print(f"NaN count: {X_full.isnull().sum().sum()}")

if X_full.shape == (1567, 591):
    print("\n✅ ALL GOOD — Ready for Phase 1")
else:
    print("\n⚠️  Shape mismatch — check files")
