# Raw Data Naming Convention

To ensure the automated pipeline can process your sensor data and correctly label the training set, all files placed in the `data/raw/` directory **MUST** follow this strict naming convention:

## Convention
`signal_{distance}_{object_name}.csv`

### Components:
1.  **`signal`**: A required static prefix for all data files.
2.  **`distance`**: The starting distance or a unique numeric identifier for the recording session.
3.  **`object_name`**: The name of the material or object being detected (e.g., `metal_plate`, `cardboard`, `person`). 
    *   *Note: This string will be used as the label for feature analysis and visualization.*
    *   *Use underscores instead of spaces (e.g., `plastic_bottle` instead of `plastic bottle`).*

---

## Examples

### ✅ Correct Filenames:
- `signal_1500_metal_plate.csv`
- `signal_2000_cardboard.csv`
- `signal_500_wall.csv`
- `signal_session1_human_subject.csv`

### ❌ Incorrect Filenames:
- `data_metal.csv` (Missing 'signal' prefix)
- `signal_metal.csv` (Missing distance/identifier part)
- `signal 1500 metal.csv` (Uses spaces instead of underscores)
- `metal_plate.csv` (Does not follow the pattern)

---

## Why is this strictly enforced?
The script `src/data/01_build_features.py` automatically scans the `data/raw/` folder. It uses the `object_name` part of the filename to automatically create a column in your dataset. This allows you to add new data and retrain your models without ever touching the code.
