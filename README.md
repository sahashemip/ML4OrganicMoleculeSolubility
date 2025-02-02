# ML4OrganicMoleculeSolubility

**Machine Learning for Predicting Solubility**

The water solubility of organic molecules is critical for optimizing the performance and stability of aqueous flow batteries, as well as for various other applications. While solubility measurements are relatively straightforward in some cases, theoretical predictions remain a significant challenge. Machine learning algorithms have become invaluable tools over the past decade to address this. High-quality data and effective descriptors are essential for building reliable, data-driven estimation models. This repository systematically investigates the effectiveness of enhanced structure-based descriptors and an outlier detection procedure to improve aqueous solubility predictability.

---

## Features
- **Enhanced Descriptors:** Novel structure-based descriptors to improve model accuracy.
- **Outlier Detection:** Robust procedures for identifying and handling data outliers.
- **Supervised Machine Learning Models:** End-to-end workflows for building, training, and evaluating solubility prediction models.

---

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:sahashemip/ML4OrganicMoleculeSolubility.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ML4OrganicMoleculeSolubility
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---

## How to Use

1. Navigate to the `notebooks` directory:
   - Open and run the Jupyter notebooks sequentially based on the numbering:
     1. `analysis`
     2. `descriptors`
     3. `ml-models`
     4. `outlier-detection`

2. **Outlier Detection**:
   - To perform outlier detection, modify the parameters in the `outlier_detector.py` script. Refer to the data in **TABLE I** of the associated manuscript for parameter details.

---

## Project Structure
- `notebooks/`: Contains step-by-step Jupyter notebooks for analysis, feature engineering, and model development.
- `scripts/`: Includes Python scripts for outlier detection and custom preprocessing utilities.
- `datasets/`: Holdes all different datasets generated by distinct descriptors.
- `outliers/`: Stores outputs related to the detected outliers.

---

## Related Scientific Work
For further details [Click here](https://chemrxiv.org/engage/chemrxiv/article-details/67851a75fa469535b9ceafbd)!

---

### Contribution Guidelines
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.
### Contribution Guidelines

### License
This project is licensed under the MIT License. See the `LICENSE` file for details.
