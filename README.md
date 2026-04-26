## 📚 Table of Contents for a perceptron in Soft Computing

1. [Repository Structure](#-repository-structure)
2. [Prerequisites & Setup](#-prerequisites--setup)
3. [How a Single Layer Perceptron Works](#-how-a-single-layer-perceptron-works)
4. [Experiments](#-experiments)
   - [Exp 1.1 – Binary Classification (AND / OR Gates)](#experiment-11--binary-classification)
   - [Exp 1.2 – Fixed Increment Learning Algorithm](#experiment-12--fixed-increment-learning-algorithm)
   - [Exp 1.3 – Convergence Analysis](#experiment-13--convergence-analysis--decision-boundary)
5. [Case Study – Medical Diagnosis](#-case-study--medical-diagnosis)
6. [Expected Outputs](#-expected-outputs)
7. [Limitations of Single Layer Perceptron](#-limitations-of-slp)
8. [Course Outcomes](#-course-outcomes)

---

## 📁 Repository Structure

```
soft-computing-lab/
│
├── experiments/
│   ├── exp1_1_binary_classification.py   # AND / OR gate classification
│   ├── exp1_2_fixed_increment.py         # Fixed Increment Learning Rule
│   └── exp1_3_convergence_analysis.py    # Convergence plot + decision boundary
│
├── case_study/
│   └── medical_diagnosis.py             # Diabetes classification (Glucose + BMI)
│
├── utils/
│   └── perceptron_utils.py              # Shared plotting & metrics helpers
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/soft-computing-lab.git
cd soft-computing-lab
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run any experiment
```bash
python experiments/exp1_1_binary_classification.py
python experiments/exp1_2_fixed_increment.py
python experiments/exp1_3_convergence_analysis.py
python case_study/medical_diagnosis.py
```

---

## 🧠 How a Single Layer Perceptron Works

A **Single Layer Perceptron (SLP)** is the fundamental building block of artificial neural networks.

### Architecture

```
        x₁ ──(w₁)──┐
        x₂ ──(w₂)──┼──► Σ (net) ──► f(net) ──► ŷ
        ...          │
        xₙ ──(wₙ)──┘
                      ↑
                    + bias (b)
```

### Forward Pass

| Symbol | Meaning |
|--------|---------|
| xᵢ     | Input features |
| wᵢ     | Learnable weights |
| b      | Bias term |
| net    | Weighted sum: `net = Σ(wᵢ · xᵢ) + b` |
| f(net) | Step activation: `1 if net ≥ 0 else 0` |

### Weight Update Rule (Learning)

```
error = target − output
Δwᵢ   = α × error × xᵢ
Δb    = α × error
```

Where **α** is the learning rate (typically 0.01 – 0.5).

### Convergence Guarantee

> **Perceptron Convergence Theorem**: If the training data is **linearly separable**,
> the perceptron algorithm is guaranteed to find a separating hyperplane in a **finite**
> number of iterations.

### Step-by-Step Working Procedure

```
1. Initialise all weights w = [0, 0, ..., 0] and bias b = 0
2. For each training epoch:
   a. For each sample (xᵢ, yᵢ):
       i.  Compute net = Σ(w · x) + b
       ii. Apply activation: ŷ = step(net)
      iii. Compute error = y − ŷ
       iv. If error ≠ 0:
             Update w ← w + α · error · x
             Update b ← b + α · error
   b. Record total misclassifications for this epoch
3. Stop when misclassifications = 0 OR max_epochs reached
4. Report final weights, bias, and decision boundary
```

---

## 🔬 Experiments

### Experiment 1.1 – Binary Classification

**File:** `experiments/exp1_1_binary_classification.py`

**Aim:** Implement SLP and verify on AND/OR logic gate datasets.

**Key Concepts:**
- Linearly separable problems
- Step activation function
- Weight initialisation & update

**Truth Tables Used:**

| x₁ | x₂ | AND | OR |
|----|----|-----|----|
| 0  | 0  |  0  |  0 |
| 0  | 1  |  0  |  1 |
| 1  | 0  |  0  |  1 |
| 1  | 1  |  1  |  1 |

**Expected Output:**
```
=== AND Gate ===
Converged at epoch 6
Weights  : [0.2 0.1]
Bias     : -0.3
Accuracy : 100.0%

=== OR Gate ===
Converged at epoch 3
Weights  : [0.2 0.2]
Bias     : -0.1
Accuracy : 100.0%
```

---

### Experiment 1.2 – Fixed Increment Learning Algorithm

**File:** `experiments/exp1_2_fixed_increment.py`

**Aim:** Train using Fixed Increment Rule and track epoch-by-epoch convergence.

**Key Concepts:**
- Fixed vs. variable learning rate
- Weight trace table
- Finite convergence

**Trace Table Output:**
```
======================================================
   Fixed Increment Perceptron – AND Gate Training
======================================================
 Epoch                  Weights    Bias   Errors
-------------------------------------------------------
     1  w=[0. 0.]  b= 0.0  err=1
     2  w=[1. 1.]  b=-1.0  err=1
     3  w=[1. 1.]  b=-1.0  err=0

✔ Converged!  Final weights: w=[1. 1.],  bias: b=-1.0
```

---

### Experiment 1.3 – Convergence Analysis & Decision Boundary

**File:** `experiments/exp1_3_convergence_analysis.py`

**Aim:** Plot the error-vs-epoch convergence curve and visualise the learned decision boundary.

**Key Concepts:**
- Decision boundary: `w₀·x₁ + w₁·x₂ + b = 0`
- Convergence epoch detection
- Matplotlib visualisation

**Expected Output:**
```
================================================
   Experiment 1.3 – Convergence Analysis
================================================
Final Weights    : [0.2 0.1]
Final Bias       : -0.3
Converged at     : epoch 6
Decision Boundary: 0.2000·x₁ + 0.1000·x₂ + -0.3000 = 0
```

Plots are saved to `experiments/exp1_3_convergence_analysis.png`.

---

## 🏥 Case Study – Medical Diagnosis

**File:** `case_study/medical_diagnosis.py`

**Problem:** Classify patients as **Diabetic (1)** or **Non-Diabetic (0)** based on:
- Glucose Level (normalised 0–1)
- BMI (normalised 0–1)

**Dataset (6 patients):**

| # | Glucose | BMI | Label | Description |
|---|---------|-----|-------|-------------|
| 1 | 0.7 | 0.8 | 1 – Diabetic | High glucose, High BMI |
| 2 | 0.2 | 0.3 | 0 – Non-Diabetic | Low glucose, Low BMI |
| 3 | 0.8 | 0.6 | 1 – Diabetic | High glucose, Medium BMI |
| 4 | 0.1 | 0.2 | 0 – Non-Diabetic | Very low glucose, Very low BMI |
| 5 | 0.6 | 0.9 | 1 – Diabetic | Medium glucose, High BMI |
| 6 | 0.3 | 0.4 | 0 – Non-Diabetic | Low glucose, Medium BMI |

**Expected Output:**
```
=================================================================
   Case Study – Medical Diagnosis (Diabetes Classification)
=================================================================
Learned Weights : Glucose = 0.400,  BMI = 0.300
Learned Bias    : -0.500
Decision Rule   : 0.400·Glucose + 0.300·BMI + -0.500 >= 0 → Diabetic

#    Glucose    BMI   Target     Pred  Status  Description
-------------------------------------------------------------------------
1        0.7    0.8        1        1  ✔ OK    High glucose, High BMI
2        0.2    0.3        0        0  ✔ OK    Low glucose, Low BMI
3        0.8    0.6        1        1  ✔ OK    High glucose, Medium BMI
4        0.1    0.2        0        0  ✔ OK    Very low glucose, Very low BMI
5        0.6    0.9        1        1  ✔ OK    Medium glucose, High BMI
6        0.3    0.4        0        0  ✔ OK    Low glucose, Medium BMI

Overall Accuracy: 100.0%
```

---

## 📊 Expected Outputs (Summary)

| Experiment | Dataset | Epochs to Converge | Accuracy |
|------------|---------|--------------------|----------|
| 1.1 – AND Gate | 4 binary samples | 6 | 100% |
| 1.1 – OR Gate  | 4 binary samples | 3 | 100% |
| 1.2 – Fixed Increment (AND) | 4 binary samples | 3 | 100% |
| 1.3 – Convergence (AND)     | 4 binary samples | 6 | 100% |
| Case Study – Diabetes | 6 patient records | < 50 | 100% |

---

## ⚠️ Limitations of SLP

| Limitation | Explanation |
|-----------|-------------|
| **Linear separability only** | SLP cannot solve the XOR problem — it requires a non-linear boundary |
| **Binary output** | The step function outputs only 0 or 1; no probability scores |
| **No hidden layers** | Cannot learn complex feature representations |
| **Sensitive to learning rate** | Too high → oscillation; too low → slow convergence |
| **Single threshold** | Cannot model multi-class problems without modification |

> **XOR Problem (unsolvable by SLP):**
> ```
> (0,0) → 0 │ (0,1) → 1
> (1,0) → 1 │ (1,1) → 0
> ```
> A straight line cannot separate XOR outputs. **Multi-Layer Perceptron (MLP)** is required.

---

*A short and detailed analysis about the working principle of a Single Layer Perceptron using techniques like Linear Summation and Thresholding, Decision Boundary Extraction and Supervised Learning Techniques*
