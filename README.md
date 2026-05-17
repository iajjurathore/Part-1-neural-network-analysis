# Name - Ajay Singh
# id - Bitsom_Ba_2511011
# Part 1: Neural Network Fundamentals and Training Behaviour Analysis

## Overview
This project builds and analyzes a **feed-forward neural network** to predict customer churn using a structured telecom dataset. The complete pipeline demonstrates how neural networks learn through forward pass, loss calculation, backpropagation, and parameter updates.

## Key Objectives

 Task 1: Dataset Understanding 
 Task 2: Data Preprocessing 
 Task 3: Neural Network Building 
 Task 4: Training & Evaluation 
 Task 5: Hyperparameter Experimentation 
 Task 6: Final Reflection 

## Dataset
- **File**: customer_churn_nn.csv
- **Location**: `https://drive.google.com/drive/folders/1Aihn49cUYMjCgeCTFBTyprjrgZO3UY6r?usp=drive_link`
- **Rows**: 2,000 samples
- **Columns**: 17 features (16 input + 1 target)
- **Target**: Binary classification (churn/retained)
- **Class Distribution**: 98.45% retained, 1.55% churned (imbalanced)

## Project Structure

```
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.png or .csv
    └── evaluation_outputs.png

```

## Key Findings

### Model Performance
- **Test Accuracy**: 98.50%
- **AUC-ROC**: 0.9850
- **Training/Test Gap**: 0.25% (excellent generalization)

### Hyperparameter Experimentation Results
Tested 7 configurations with variations in:
- Architecture (depth/width)
- Learning rate (0.0001 → 0.01)
- Activation functions (ReLU vs Tanh)
- Batch sizes

**Best Configuration**: Baseline (64→32 neurons, ReLU, LR=0.001)

## Technologies Used

- **Framework**: TensorFlow 2.15 + Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Optimization**: Adam optimizer with early stopping

## Model Architecture

```
Input Layer (24 neurons)
          ↓
Dense Layer 1 (64 neurons, ReLU)
          ↓
Dense Layer 2 (32 neurons, ReLU)
          ↓
Output Layer (1 neuron, Sigmoid)
          ↓
Binary Classification (Churn/Retained)
```

## Recommendations for Improvement

1. Implement `class_weight='balanced'` to handle class imbalance
2. Use F1-Score or AUC-ROC as primary metrics (not accuracy)
3. Apply SMOTE for synthetic minority oversampling
4. Experiment with threshold adjustment (0.5 → 0.3-0.4)
5. Monitor both majority and minority class metrics separately


## References
- TensorFlow/Keras Documentation
- Understanding Neural Network Training Dynamics
- Class Imbalance in Machine Learning


