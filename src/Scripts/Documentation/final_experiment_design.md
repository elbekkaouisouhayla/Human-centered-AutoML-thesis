# Final Experimental Design

## Study A: Controlled Predictive Benchmark

Frameworks:
- PyCaret
- TPOT
- H2O AutoML

Datasets:
- Breast Cancer Wisconsin Diagnostic
- Wine Recognition
- Titanic

Seeds:
- 42
- 123
- 2026

Train-test split:
- 80% training
- 20% testing
- Stratified by target

Cross-validation:
- 5 folds

External preprocessing:
- Same preprocessing for all three frameworks
- Fitted on training data only
- Applied to test data afterward

Principal metric:
- Macro-averaged F1

Additional metrics:
- Accuracy
- Macro precision
- Macro recall
- ROC AUC

Runtime:
- Maximum 10 minutes where supported
- Actual runtime recorded in seconds

Total experiments:
3 datasets × 3 frameworks × 3 seeds = 27 runs


## Study B: Native Automation and Scalability Case Study

Dataset:
- Home Credit application_train.csv

Purpose:
- Native missing-value handling
- Native categorical-variable handling
- Data preparation automation
- Scalability
- Runtime behaviour
- Error behaviour
- Manual intervention required

Important:
- Home Credit is not combined with the controlled predictive averages.
- It is reported as a separate case study.


## Qualitative Evaluation

### Automation
1. Raw-data acceptance and preparation
2. Missing-value handling
3. Categorical-feature handling
4. Model search
5. Hyperparameter optimization
6. Ensembling and final selection

### Usability
1. Installation and initialization
2. Workflow simplicity
3. Output and error clarity
4. Documentation accessibility

### Transparency and Explainability
1. Preprocessing and pipeline visibility
2. Candidate and search visibility
3. Final model and parameter visibility
4. Global explanation accessibility

### User Control and Configurability
1. Pre-run configuration
2. Progress visibility and interruption
3. Model selection and modification
4. Export and reuse

Total indicators:
18 indicators × 3 frameworks = 54 scores


## Scoring Scale

0 = unavailable or unsuccessful under the defined conditions
1 = available indirectly or requiring substantial additional work
2 = directly available but with important limitations
3 = directly available, clearly documented and practically usable


## Evidence Sources

Each score must be supported by at least one of:
- Notebook output
- Log file
- Screenshot
- Exported pipeline
- Saved model
- Framework object
- Error message
- Official documentation