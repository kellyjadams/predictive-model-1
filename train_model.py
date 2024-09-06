# import libraries
import pickle
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, make_scorer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imbalance_pipeline
import pandas as pd
import numpy as np
import os

# Load data
base_path = 'C:\\Users\\MyUser\\Projects\\'
filename = os.path.join(base_path, 'model_data.csv')
df = pd.read_csv(filename)

# Convert 'month' from string to datetime format for sorting purposes
df['month'] = pd.to_datetime(df['month'])

# Pivot the DataFrame to wide format
df_wide = df.pivot_table(index='user_id', columns='month', values=['num_logins', 'games_played', 'total_purchases'], aggfunc='sum')

# Fill NaN values that occur from pivoting (no activity months will be NaN)
df_wide.fillna(0, inplace=True)

# Create trend and average features
# Calculating monthly changes as trends for each activity
for activity in ['num_logins', 'games_played', 'total_purchases']:
    df_wide[(activity, 'trend')] = df_wide[activity].diff(axis=1).mean(axis=1)
    df_wide[(activity, 'average')] = df_wide[activity].mean(axis=1)

# Assign churn or not based on activity in the last month (assumed: no activity = churn)
last_month = max(df['month'])
df_wide['churned'] = (df_wide[('num_logins', last_month)] + df_wide[('games_played', last_month)] + df_wide[('total_purchases', last_month)] == 0).astype(int)

# Flatten the columns after wide transformation
df_wide.columns = ['_'.join(col).strip() for col in df_wide.columns.values]

# Now df_wide is ready for feature selection and model training
print(df_wide.head())

# Model

# Select final features for the model; use corrected DataFrame and column names
final_feature_set = [col for col in df_wide.columns if 'trend' in col or 'average' in col]  # modify as needed
X = df_wide[final_feature_set]
y = df_wide['churned']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Address class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Set up a pipeline with scaling and random forest
pipeline = make_imbalance_pipeline(
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

# Perform cross-validation on the resampled training data
cv_scores = cross_val_score(pipeline, X_train_resampled, y_train_resampled, cv=5, scoring=make_scorer(roc_auc_score))
print(f'Cross-Validation AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# Tune the hyperparameters using GridSearchCV
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [None, 10, 20, 30],
    'randomforestclassifier__min_samples_split': [2, 5, 10],
    'randomforestclassifier__min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_resampled, y_train_resampled)

# Print the best parameters and the corresponding AUC score
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation AUC: {grid_search.best_score_:.4f}')

# Finalize the model by training on the entire dataset
pipeline.set_params(**grid_search.best_params_)
pipeline.fit(X_train_resampled, y_train_resampled)

# Evaluate on the hold-out test set using optimal threshold
test_probs = pipeline.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
optimal_idx = np.argmax(precision * recall)  # Example to maximize F1
optimal_threshold = thresholds[optimal_idx]
test_predictions = (test_probs >= optimal_threshold).astype(int)
test_auc = roc_auc_score(y_test, test_probs)
print(f'Optimal Threshold: {optimal_threshold}')
print(f'Test Set AUC: {test_auc:.4f}')

# Save the finalized model
model_filename = os.path.join(base_path, 'model_rf.pkl')
joblib.dump(pipeline, model_filename)

# Save the feature set used
feature_names_filename = os.path.join(base_path, 'feature_names_model_rf.pkl')
with open(feature_names_filename, 'wb') as f:
    pickle.dump(final_feature_set, f)

print(f'Model and feature names saved as {model_filename} and {feature_names_filename}.')
