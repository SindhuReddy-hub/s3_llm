import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from datetime import datetime

# Load dataset
df = pd.read_csv("s3_bucket_deletion_dataset.csv")

# Feature Engineering
df['creation_date'] = pd.to_datetime(df['creation_date'])
df['bucket_age_days'] = (datetime.now() - df['creation_date']).dt.days

# Encode categorical variables
le_model = LabelEncoder()
le_env = LabelEncoder()
le_lifecycle = LabelEncoder()

df['s3_model_version_encoded'] = le_model.fit_transform(df['s3_model_version'])
df['operating_env_encoded'] = le_env.fit_transform(df['operating_env'])
df['lifecycle_encoded'] = le_lifecycle.fit_transform(df['core_bud_module_lifecycle'])

# Final feature set
X = df[['total_objects', 's3_model_version_encoded', 'operating_env_encoded', 'lifecycle_encoded', 'bucket_age_days']]
y = df['can_delete']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# --- 🧪 Predict on New Bucket Metadata ---
new_buckets = pd.DataFrame([
    {
        "total_objects": 0,
        "s3_model_version": "No Access 9.5",
        "operating_env": "dev",
        "core_bud_module_lifecycle": "Archived",
        "creation_date": "2023-01-01"
    },
    {
        "total_objects": 2000,
        "s3_model_version": "Access 10.0",
        "operating_env": "prod",
        "core_bud_module_lifecycle": "Prohibited",
        "creation_date": "2022-09-01"
    }
])

# Feature engineering for new data
new_buckets['creation_date'] = pd.to_datetime(new_buckets['creation_date'])
new_buckets['bucket_age_days'] = (datetime.now() - new_buckets['creation_date']).dt.days
new_buckets['s3_model_version_encoded'] = le_model.transform(new_buckets['s3_model_version'])
new_buckets['operating_env_encoded'] = le_env.transform(new_buckets['operating_env'])
new_buckets['lifecycle_encoded'] = le_lifecycle.transform(new_buckets['core_bud_module_lifecycle'])

X_new = new_buckets[['total_objects', 's3_model_version_encoded', 'operating_env_encoded', 'lifecycle_encoded', 'bucket_age_days']]
preds = model.predict(X_new)
probs = model.predict_proba(X_new)[:, 1]

# Show results
new_buckets['can_delete'] = preds
new_buckets['confidence'] = probs.round(2)
print("\n🔮 Prediction Results:")
print(new_buckets[['total_objects', 's3_model_version', 'operating_env', 'can_delete', 'confidence']])
