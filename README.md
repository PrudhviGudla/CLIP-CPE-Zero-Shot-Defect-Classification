# CLIP-CPE-Zero-Shot-Defect-Classification
Zero-Shot Multi-Class Defect Classification pipeline via CLIP Model using Compositional Prompt Ensemble Technique from WinCLIP, tested on MVTec AD Hazelnut Dataset. It can identify and classify various types of defects without requiring any training data, making it highly practical for industrial deployment where labeled defect data is scarce or expensive to obtain.

## Method
### Core Architecture
The system employs the CLIP vision-language model along with CPE. Compositional Prompt Ensemble (CPE) creates diverse text descriptions for each class using multiple templates and state words

**Two-Stage Classification Pipeline:**

- Stage 1: Binary classification (Normal vs Anomalous) 

- Stage 2: Multi-class defect type classification (only for detected anomalies)

Other Details
- Base Model: OpenAI CLIP (ViT-L/14)
- Binary Classification Threshold is set to 0. It can be set to the optimal threshold calculated from the ROC curve for better results.
- Feature Extraction: Global image features through CLIP 
- Prompt Templates: Multiple engineered templates for robust classification
- State Words: Carefully curated descriptive terms for each defect class

Compositional Prompt Ensemble
The system generates multiple text prompts for each class:

```
# Example for "crack" defect

Templates: ["a photo of {}", "an image showing {}", "this looks like {}"]

States: ["cracked hazelnut", "hazelnut with crack", "damaged hazelnut"]

```
Results in 9 different prompts for "crack" class

## 📊 Results

### Two-Stage Classification (Binary + Multi-Class)
Binary Classification Threshold set to 0
| Metric | Binary (Normal vs Anomaly) | Multi-Class (All Classes) |
|--------|---------------------------:|---------------------------:|
| **Overall Accuracy** | **78.2%** | **64.5%** |
| **Mean Precision** | **0.797** | **0.750** |
| **Mean Recall** | **0.818** | **0.560** |
| **Mean F1-Score** | **0.780** | **0.530** |
| **ROC AUC** | **0.9004** | **N/A** |

#### Per-Class Performance (Multi-Class):
| Class | Precision | Recall | F1-Score | Accuracy |
|-------|----------:|-------:|---------:|---------:|
| **Good** | 0.633 | 0.950 | 0.760 | 95.0% |
| **Crack** | 0.538 | 0.778 | 0.636 | 77.8% |
| **Cut** | 0.800 | 0.235 | 0.364 | 23.5% |
| **Hole** | 0.778 | 0.778 | 0.778 | 77.8% |
| **Print** | 1.000 | 0.059 | 0.111 | 5.9% |

### Single-Stage Direct Classification

| Metric | Multi-Class (All Classes) |
|--------|--------------------------:|
| **Overall Accuracy** | **38.2%** |
| **Mean Precision** | **0.408** |
| **Mean Recall** | **0.476** |
| **Mean F1-Score** | **0.365** |

### 🔍 Key Insights

- **Two-stage approach significantly outperforms single-stage** (64.5% vs 38.2% accuracy)
- **Binary classification shows strong performance** (78.2% accuracy), indicating good normal/anomaly separation  
- **Cut and Print defects are most challenging** with low recall rates, requiring further optimization

The results demonstrate the effectiveness of the two-stage pipeline over direct multi-class classification for zero-shot defect detection in hazelnuts.

## Usage
Installation
```
git clone https://github.com/PrudhviGudla/CLIP-CPE-Zero-Shot-Defect-Classification.git
cd CLIP-CPE-Zero-Shot-Defect-Classification
pip install -r requirements.txt
```

### Method 1: Using the Python Script
```
python run_inference.py --dat_dir /path/to/test/data 
```

### Method 2: Using Jupyter Notebook
- Open and run WinCLIP-Zero-Shot-Defect-Classification.ipynb by following the instructions in the notebook
- You can run different classification experiments
- View comprehensive performance metrics and visualizations

Dataset Structure
Organize your test data as follows:
```
test/
├── good/
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── crack/
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── cut/
├── hole/
└── print/
```

### Configuration
Modify spec.py's DefectClassificationSpec Pydantic class to customize the model, prompt templates and state words for different classes.

```
class DefectClassificationSpec:
    model_name = "ViT-L/14"
    
    # Customize templates
    templates = [
        "a photo of {}",
        "an image showing {}",
        # Add more templates
    ]
    
    # Customize state words for each class
    normal_states = ["good hazelnut", "perfect hazelnut"]
    anomalous_states = ["defective hazelnut", "damaged hazelnut"]
    
    defect_states = {
        "crack": ["cracked hazelnut", "hazelnut with crack"],
        "cut": ["cut hazelnut", "hazelnut with cut"],
        # Add more defect types
    }
```

## Acknowledgments
- WinCLIP Paper: Based on "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation" (CVPR 2023)
- OpenAI CLIP
- MVTec AD Dataset
