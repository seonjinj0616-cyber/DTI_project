import h5py
import os

BASE_DIR = r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB\final_final"

FILES = [
    "IC50_vectorDB.h5",
    "Kd_vectorDB.h5",
    "Ki_vectorDB.h5",
]

def print_h5(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(
            f"  DATASET : {name}"
            f" | shape={obj.shape}"
            f" | dtype={obj.dtype}"
        )
    elif isinstance(obj, h5py.Group):
        print(f"  GROUP   : {name}")


for filename in FILES:

    path = os.path.join(BASE_DIR, filename)

    print("\n" + "=" * 100)
    print(filename)
    print("=" * 100)

    if not os.path.exists(path):
        print("❌ FILE NOT FOUND")
        continue

    with h5py.File(path, "r") as f:

        print("\n[ROOT ATTRIBUTES]")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")

        print("\n[H5 STRUCTURE]")
        f.visititems(print_h5)