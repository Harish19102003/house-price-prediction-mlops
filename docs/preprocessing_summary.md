## 🛠️ Preprocessing Module Summary

The preprocessing module handles initial data preparation steps for the ML pipeline. It ensures clean, high-quality input data for model training and evaluation.

### Features:
- **Data Loading:**  
  Loads raw CSV datasets from the `data/raw/` directory using the `load_raw_data()` function, with error handling if files are missing.

- **Missing Value Handling:**  
  Fills missing values intelligently:
  - Numerical columns are filled with their **median**.
  - Categorical columns are filled with their **mode** (most frequent value).

- **Processed Data Saving:**  
  Saves the cleaned and preprocessed dataset to `data/processed/` using the `save_processed_data()` function, ensuring directories are created if they don't exist.

### Functions:
| Function Name        | Purpose                                             |
| -------------------- | ---------------------------------------------------- |
| `load_raw_data()`     | Loads the dataset from a given file path.             |
| `fill_missing_values()` | Automatically fills missing values in the dataset. |
| `save_processed_data()` | Saves the processed DataFrame to disk.             |

---

✅ **Current Status:**  
Preprocessing pipeline is fully tested and operational in the Jupyter notebook.
