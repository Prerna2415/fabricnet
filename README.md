# FabricNet-Inspired Fabric Composition Recognition

A deep learning system for predicting fiber composition from fabric images at the patch level, inspired by FabricNet. Given a fabric image patch, the model outputs the estimated proportion of each of six fiber types present — rather than a single fabric class label.

Built during a Research Internship at CDSAML, PES University.

---

## Results

| | Accuracy |
|---|---|
| Original FabricNet paper | 84% |
| **This implementation (modified architecture, custom dataset)** | **87%** |

The 3-point improvement comes from architectural modifications to the FabricNet backbone and multi-patch feature integration, evaluated on a custom-collected dataset rather than the original paper's dataset.
<img width="753" height="240" alt="image" src="https://github.com/user-attachments/assets/8d80ecee-60dd-46a1-b21b-07f87d21a2af" />


---

## Architecture

A compact Xception-style convolutional backbone feeding six parallel composition-prediction heads:

<img width="857" height="816" alt="image" src="https://github.com/user-attachments/assets/75a1f7fd-506c-4359-8e5a-e2b1641fa09d" />

- **Input:** 120×120 RGB fabric patches
- **Backbone:** entry-flow-style separable convolutions with residual (projection shortcut) connections — a lighter-weight variant of the Xception design used in the original FabricNet
- **Output:** 6 independent sigmoid heads, each predicting the composition fraction of one fiber type from the patch, allowing multi-fiber blends to be represented (rather than a single hard class label)

*(Fill in: which six fiber types — e.g. cotton, polyester, wool, silk, nylon, linen — and whether patch-level predictions are aggregated to a garment-level composition estimate downstream.)*

---

## Training

- **Validation strategy:** k-fold cross-validation — this repo's checkpoint (`fabricnet_best_fold_3.keras`) is the best-performing checkpoint from fold 3.
- **Loss / metrics:** *(fill in — e.g. per-head binary cross-entropy or MSE, and how the 84%/87% accuracy figure is derived from six regression outputs)*
- **Ensembling considered:** ensemble methods, InceptionResNetV2, and Vision Transformer backbones were evaluated as alternatives/additions during benchmarking.

---

## Project Structure

```
.
├── data/                       # fabric patch dataset (custom-collected)
├── models/
│   └── fabricnet_best_fold_3.keras
├── training/                   # training + k-fold CV scripts
├── evaluation/                 # accuracy / metric computation
├── inference.py                # load model, predict composition on new patches
└── README.md
```
*(adjust to match your actual repo layout)*

---

## Usage

```python
import keras

model = keras.saving.load_model("fabricnet_best_fold_3.keras")

# patch: numpy array, shape (1, 120, 120, 3), pixel values normalized as in training
predictions = model.predict(patch)
# predictions -> list of 6 arrays, each shape (1, 1): composition fraction per fiber type
```

*(fill in your actual preprocessing/normalization steps — this matters for correct inference)*

---

## Status

Model trained and evaluated; current best checkpoint (fold 3) achieves 87% accuracy on the custom dataset, exceeding the original FabricNet paper's 84%.

## Roadmap

- [ ] *(e.g. patch-to-garment aggregation)*
- [ ] *(e.g. expand fiber type coverage)*
- [ ] *(e.g. deployment / inference API)*

---

## Reference

Inspired by FabricNet (original paper), adapted with architectural modifications and trained on a custom-collected dataset as part of a Research Internship at CDSAML, PES University.
