# ML-based prediction of elastic properties using reduced datasets of accurate calculations results


This repository contains data, described in the paper *"Machine learning-based prediction of elastic properties using reduced datasets of accurate calculations results"* 

---

# Files
- `model_parametes.info` - used parameters for model training
- `custom_model.py` - python library containing the model as a class object
- `multimodel.py` - script for using a pre-trained model
- `test.txt` - example input file for the `multimodel.py`
- `data/db.feats.pt` - initial train set, with pre-calculated features, described in the article
- `data/Default_PAW_potentials_VASP.csv` - file used to calculate features to use the model
- `data/hull_feats.json` - file used to calculate features to use the model
- `data/model.dump` - pretrained model

---

The main concept of this work is creation of two stacked estimators trained in a specific way. The first one is trained on large datatset of less accurate calculations made using EMTO-CPA and the second is trained on the much smaller dataset of more accurate PAW-SQS calculations.

---

# Usage

  - Ensure that all requirements reached (see `requirements.txt`
  - `python multimodel.py -i test.txt`
  - output will be saved as `[input filename].csv`
