from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepare_train_test(df, label='no_show_num', test_size=0.3, random_state=42):
    X = df.drop(columns=['no_show', label])
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    sm = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    return X_train_bal, X_test, y_train_bal, y_test
