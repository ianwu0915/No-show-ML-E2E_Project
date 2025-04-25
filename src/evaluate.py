import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def evaluate_model(y_test, y_pred, y_pred_proba, model_name="Model"):
    """評估模型性能"""
    # 計算混淆矩陣
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # 計算各項指標
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)  # 添加accuracy計算
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    plot_confusion_matrix(y_test, y_pred, model_name)
    print("--------------------------------")
    print(classification_report(y_test, y_pred))
    print("roc_auc: ", roc_auc)
    
    return metrics

def evaluate_nn(model, test_loader, device, threshold=0.5):
    """Evaluate the model with test data given a threshold"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.reshape(-1, 1) 
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
            
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)
    y_true = np.array(all_labels)
    
    return y_pred, y_proba, y_true

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    繪製混淆矩陣，並明確標示 show/no-show
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 8))
    
    plt.subplot(111)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    tick_labels = ['Show (0)', 'No-show (1)']
    plt.xticks([0.5, 1.5], tick_labels)
    plt.yticks([0.5, 1.5], tick_labels)
    
    plt.title(f'Confusion Matrix for {model_name}\n')
    
    # 修改文字格式，使用單行字符串並調整對齊
    tn, fp, fn, tp = cm.ravel()
    explanation_text = (
        f"True Negatives (TN) = {tn}: Predict No-show, Actual No-show\n"
        f"False Positives (FP) = {fp}: Predict No-show, Actual Show\n"
        f"False Negatives (FN) = {fn}: Predict Show, Actual No-show\n"
        f"True Positives (TP) = {tp}: Predict Show, Actual Show\n\n"
        f"No-show Recall = {tp/(tp+fn):.3f}: Correctly identify the proportion of patients who will not show up"
    )
    
    plt.figtext(0.02, 0.02, explanation_text, fontsize=10, va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    plt.show()

def evaluate_threshold(y_test, y_proba, thresholds=None):
    """
    Evaluate effects of different thresholds
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    thresholds : array-like, optional
        List of thresholds to test, defaults to np.arange(0.1, 0.5, 0.05)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.5, 0.05)
    
    results = []
    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        
        # Calculate confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
        
        # Calculate business metrics
        no_show_cost = 200  # Cost of undetected no-show
        intervention_cost = 20  # Cost of intervention
        
        # Calculate cost savings
        cost_savings = (tp * no_show_cost) - ((fp + tp) * intervention_cost)
        
        results.append({
            'threshold': threshold,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'recall': tp / (tp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'cost_savings': cost_savings
        })
    
    results_df = pd.DataFrame(results)
    print("Best threshold based on cost savings:", 
          results_df.loc[results_df['cost_savings'].idxmax(), 'threshold'])
    
    return results_df

def plot_feature_importance_xgb(model, top_n=20):
    import xgboost as xgb
        
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, 
                    max_num_features=top_n,
                    importance_type='weight',
                    height=0.5,
                    title='Feature Importance - XGBoost')
    plt.show()
