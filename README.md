\# Fashion MNIST Classification — MLP from Scratch



Implementation of a Multi-Layer Perceptron (MLP) built entirely from scratch using NumPy, trained on the Fashion MNIST dataset. No deep learning frameworks were used for the model itself.



\## Results



\- Test Accuracy: \*\*\~86%\*\*

\- Epochs: 20

\- Batch Size: 64



\## Network Architecture



| Layer | Size | Activation |

|-------|------|------------|

| Input | 784 | — |

| Hidden | 128 | Sigmoid |

| Output | 10 | Sigmoid |



\## Implementation Details



\- \*\*Weight Initialization:\*\* Xavier (Glorot)

\- \*\*Loss Function:\*\* Cross-entropy

\- \*\*Optimizer:\*\* Mini-batch Gradient Descent (lr=0.1)

\- \*\*Backpropagation:\*\* Implemented manually using chain rule



\## Technologies



\- Python

\- NumPy

\- TensorFlow (dataset loading only)

\- Matplotlib



\## Files



\- `main.py` — Full training and evaluation script

\- `notebook.ipynb` — Jupyter notebook version

\- `report.pdf` — Detailed project report



\## Usage



```bash

pip install numpy tensorflow matplotlib

python main.py

```

