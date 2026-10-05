# Titanic Survival Prediction (Logistic Regression)

Predicting which Titanic passengers survived, using Python and Scikit-learn.
Built as part of my Data Science with AI learning journey, following a simple, step-by-step style: **Look → Clean → Split → Train → Evaluate → Explain → Predict.**

| Test accuracy | Baseline ("everyone died") | Kaggle score |
|---|---|---|
| **80.4%** | 61.5% | **0.76794** |

## Problem
Supervised learning, binary classification: predict `Survived` (0/1) from passenger details such as class, sex, age and family size.
Dataset: [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) (891 training passengers, 418 test passengers).

## Approach
1. **Explore:** shape, nulls, duplicates, survival rate by sex and class.
2. **Clean:**
   - Checked duplicates *before* dropping the ID column (otherwise real passengers get deleted).
   - Dropped `PassengerId`, `Name`, `Ticket`, `Cabin` (identifiers, or about 77% empty).
   - Filled `Age` with the median and `Embarked` with the mode.
   - Checked `Fare` outliers with IQR and **kept them**, since they are genuine 1st-class fares.
3. **Encode:** one-hot encoding for `Sex` and `Embarked`.
4. **Split:** stratified 80/20 train-test split (`random_state=42`).
5. **Model:** Logistic Regression.
6. **Evaluate:** accuracy, confusion matrix and classification report, compared against a baseline.
7. **Predict:** applied the same cleaning to the test set (using train medians) and generated `submission.csv`.

## Key findings
- **Sex** was the strongest factor: men were far less likely to survive.
- **Ticket class** mattered: 3rd class had lower survival odds than 1st.
- Younger passengers had a slightly better chance.

## Results
![Results](images/titanic_4_results.png)

## Limitations and next steps
- The model is linear and misses combinations (for example a 3rd-class woman travelling alone).
- Local accuracy (80.4%) is higher than the Kaggle score (0.768) because the local test split is small (179 passengers).
- Planned improvements: extract titles (Mr, Mrs, Miss, Master) from `Name`, add `FamilySize` and `IsAlone`, try Random Forest, and use 5-fold cross-validation.

## How to run
```bash
git clone https://github.com/ravisurtani/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
# download train.csv and test.csv from Kaggle into this folder
jupyter notebook Titanic_Logistic_Regression.ipynb
```

## Tech stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter

## Files
- `Titanic_Logistic_Regression.ipynb`: full notebook with outputs
- `submission.csv`: Kaggle predictions (418 rows)
- `images/`: project visuals
