import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_training_history(history):
    """繪製訓練過程圖表"""
    plt.figure(figsize=(15, 5))
    
    # 損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 評估指標
    plt.subplot(1, 2, 2)
    metrics_df = pd.DataFrame(history['val_metrics'])
    plt.plot(metrics_df['f1'], label='F1 Score')
    plt.plot(metrics_df['precision'], label='Precision')
    plt.plot(metrics_df['recall'], label='Recall')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_threshold_analysis(results_df):
    """繪製閾值分析圖表"""
    plt.figure(figsize=(15, 10))
    
    # 閾值與F1分數的關係
    plt.subplot(2, 2, 1)
    plt.plot(results_df['threshold'], results_df['f1'])
    plt.title('F1 Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    
    # 閾值與精確率/召回率的關係
    plt.subplot(2, 2, 2)
    plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
    plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
    plt.title('Precision and Recall vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    
    # 閾值與成本節約的關係
    plt.subplot(2, 2, 3)
    plt.plot(results_df['threshold'], results_df['cost_savings'])
    plt.title('Cost Savings vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Savings ($)')
    
    # 閾值與TP/FP的關係
    plt.subplot(2, 2, 4)
    plt.plot(results_df['threshold'], results_df['true_positives'], label='True Positives')
    plt.plot(results_df['threshold'], results_df['false_positives'], label='False Positives')
    plt.title('True Positives vs False Positives')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png')
    plt.show()

def plot_roc_pr_curves(y_true, y_proba):
    """繪製ROC曲線和PR曲線"""
    plt.figure(figsize=(12, 5))
    
    # ROC曲線
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    
    # PR曲線
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    plt.tight_layout()
    plt.savefig('roc_pr_curves.png')
    plt.show()

def visualize_threshold_results(results_df):
    from matplotlib import pyplot as plt
    # 評估不同閾值

    # 視覺化結果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 繪製precision-recall vs threshold
    ax1.plot(results_df['threshold'], results_df['precision'], label='Precision')
    ax1.plot(results_df['threshold'], results_df['recall'], label='Recall')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision and Recall vs Threshold')
    ax1.legend()
    ax1.grid(True)

    # 繪製cost savings vs threshold
    ax2.plot(results_df['threshold'], results_df['cost_savings'])
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Cost Savings ($)')
    ax2.set_title('Cost Savings vs Threshold')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # 打印最佳閾值的詳細結果
    best_threshold_idx = results_df['cost_savings'].idxmax()
    best_result = results_df.loc[best_threshold_idx]

    print("\nBest Threshold Results:")
    print(f"Threshold: {best_result['threshold']:.3f}")
    print(f"Precision: {best_result['precision']:.3f}")
    print(f"Recall: {best_result['recall']:.3f}")
    print(f"Cost Savings: ${best_result['cost_savings']:,.2f}")
    print(f"True Positives: {best_result['true_positives']}")
    print(f"False Positives: {best_result['false_positives']}")
    print(f"False Negatives: {best_result['false_negatives']}")