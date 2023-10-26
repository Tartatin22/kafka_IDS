import random
from joblib import dump

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# Generate train data
X, y = fetch_kddcup99(
    subset="SA", percent10=True, random_state=42, return_X_y=True, as_frame=True
)
y = (y != b"normal.").astype(np.int32)
X, X_test, y, y_test = train_test_split(X, y, train_size=0.1, stratify=y, random_state=42)

# transform categorical columns into features
cat_columns = ["protocol_type", "service", "flag"]
ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
        )
preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", ordinal_encoder, cat_columns),
            ],
        remainder="passthrough",
        )
clf = IsolationForest()
pipeline = make_pipeline(preprocessor, clf)

# train the model
pipeline.fit(X)

y_pred = pipeline.predict(X_test)

dump(pipeline, './model/isolation_forest.joblib')

