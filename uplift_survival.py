import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from causalml.inference.meta import BaseXClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load dataset and data dictionary
df = pd.read_csv("./data/turnover.csv", encoding='latin-1').drop(["greywage", "novator"], axis=1)

# Preprocessing based on data dictionary
# Convert categorical features
categorical_cols = ['gender', 'industry', 'profession', 'traffic', 'head_gender', 'way', 'coach']
# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform the categorical columns
encoded_data = encoder.fit_transform(df[categorical_cols])

# Create a DataFrame with the encoded data and appropriate column names
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

# Drop original categorical columns and concatenate the encoded columns
df = df.drop(columns=categorical_cols)
df = pd.concat([df, encoded_df], axis=1)

# Create synthetic treatment variable (30% of employees receive intervention)
np.random.seed(42)
treatment_proba = 0.3 + 0.4 * (df['extraversion']/10)  # Higher extraversion more likely to get treatment
df['treatment'] = np.random.binomial(1, treatment_proba)

# Prepare survival analysis variables
df.rename(columns={'stag': 'tenure'}, inplace=True)
df['event'] = df['event'].astype(int)

# Feature scaling
scaler = StandardScaler()
features = ['age', 'extraversion', 'independ', 'selfcontrol', 'anxiety'] + list(encoded_df.columns)
df[features] = scaler.fit_transform(df[features])

# Split data
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Uplift modeling with corrected learner types
uplift_model = BaseXClassifier(
    control_outcome_learner=RandomForestClassifier(n_estimators=100, random_state=42),
    treatment_outcome_learner=RandomForestClassifier(n_estimators=100, random_state=42),
    control_effect_learner=RandomForestRegressor(n_estimators=100, random_state=42),
    treatment_effect_learner=RandomForestRegressor(n_estimators=100, random_state=42),
    control_name=0,
)

# 1. Calculate propensity scores using ORIGINAL DataFrame
propensity_model = LogisticRegression(max_iter=1000, random_state=42)

# Use original training features (before preprocessing)
propensity_features = train.drop(['treatment', 'event', 'tenure'], axis=1)
propensity_model.fit(propensity_features, train['treatment'])

# Get propensity scores from ORIGINAL features
train_propensity = propensity_model.predict_proba(propensity_features)[:, 1]

# 2. Pass propensity scores to fit()
uplift_model.fit(
    X=train,
    treatment=train['treatment'],
    y=train['event'],
    p=train_propensity  # Explicit propensity scores
)


# 3. Calculate test propensity scores
test_propensity = propensity_model.predict_proba(
    test.drop(['treatment', 'event', 'tenure'], axis=1)
)[:, 1]

# Add uplift to TRAIN data
train['uplift'] = uplift_model.predict(
    X=train,
    p=train_propensity  # Use training propensity scores
)

# 4. Predict with test features and propensity scores
test['uplift'] = uplift_model.predict(
    X=test,
    p=test_propensity
)

# Survival analysis with Cox model
cph_features = features + ['uplift']

cph = CoxPHFitter(penalizer=0.01)
cph.fit(train[cph_features + ['tenure', 'event']],
        duration_col='tenure',
        event_col='event')

# Results summary
print("Cox Model Summary:")
print(cph.summary)
print(f"\nConcordance Index: {cph.concordance_index_:.3f}")

# Visualize feature impacts
plt.figure(figsize=(10, 20))
cph.plot(hazard_ratios=True)
plt.title('Hazard Ratios for Employee Turnover Predictors')
plt.show()
