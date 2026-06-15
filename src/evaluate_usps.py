import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf

# Adjust path to import loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data, load_usps

def main():
    os.makedirs("reports", exist_ok=True)
    
    # Load data
    print("Loading datasets...")
    X_train, X_val, y_train, y_val, X_test = load_data(data_dir="data")
    X_usps_train, y_usps_train, X_usps_test, y_usps_test = load_usps(data_dir="data")
    
    # Prepare tf datasets
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(128)
    usps_ds_test = tf.data.Dataset.from_tensor_slices((X_usps_test, y_usps_test)).batch(128)
    
    # Load models
    print("\nLoading models from models/ directory...")
    cnn = tf.keras.models.load_model("models/improved_cnn.keras")
    lenet = tf.keras.models.load_model("models/lenet5.keras")
    resnet = tf.keras.models.load_model("models/mini_resnet.keras")
    
    with open("models/svm_model.pkl", "rb") as f:
        svm = pickle.load(f)
    with open("models/pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
        
    results = {}
    
    # 1. Evaluate Improved CNN
    print("\nEvaluating Improved CNN...")
    _, cnn_val_acc = cnn.evaluate(val_ds, verbose=0)
    _, cnn_usps_acc = cnn.evaluate(usps_ds_test, verbose=0)
    cnn_preds = np.argmax(cnn.predict(val_ds, verbose=0), axis=1)
    results['Improved CNN'] = {'accuracy': cnn_val_acc, 'usps_accuracy': cnn_usps_acc, 'train_time': 1151.28, 'preds': cnn_preds}
    
    # 2. Evaluate LeNet-5
    print("Evaluating LeNet-5...")
    _, lenet_val_acc = lenet.evaluate(val_ds, verbose=0)
    _, lenet_usps_acc = lenet.evaluate(usps_ds_test, verbose=0)
    lenet_preds = np.argmax(lenet.predict(val_ds, verbose=0), axis=1)
    results['LeNet-5'] = {'accuracy': lenet_val_acc, 'usps_accuracy': lenet_usps_acc, 'train_time': 111.18, 'preds': lenet_preds}
    
    # 3. Evaluate Mini-ResNet
    print("Evaluating Mini-ResNet...")
    _, resnet_val_acc = resnet.evaluate(val_ds, verbose=0)
    _, resnet_usps_acc = resnet.evaluate(usps_ds_test, verbose=0)
    resnet_preds = np.argmax(resnet.predict(val_ds, verbose=0), axis=1)
    results['Mini-ResNet'] = {'accuracy': resnet_val_acc, 'usps_accuracy': resnet_usps_acc, 'train_time': 1879.30, 'preds': resnet_preds}
    
    # 4. Evaluate SVM
    print("Evaluating SVM...")
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_val_pca = pca.transform(X_val_flat)
    svm_preds = svm.predict(X_val_pca)
    svm_val_acc = accuracy_score(y_val, svm_preds)
    
    X_usps_test_flat = X_usps_test.reshape(X_usps_test.shape[0], -1)
    X_usps_test_pca = pca.transform(X_usps_test_flat)
    usps_svm_preds = svm.predict(X_usps_test_pca)
    svm_usps_acc = accuracy_score(y_usps_test, usps_svm_preds)
    results['SVM'] = {'accuracy': svm_val_acc, 'usps_accuracy': svm_usps_acc, 'train_time': 79.60, 'preds': svm_preds}
    
    # Summary print
    print("\n" + "="*50)
    print("EVALUATION RESULTS:")
    print("="*50)
    for name in ['Improved CNN', 'LeNet-5', 'Mini-ResNet', 'SVM']:
        res = results[name]
        print(f"{name:15s} | MNIST Val: {res['accuracy']*100:6.2f}% | USPS Test: {res['usps_accuracy']*100:6.2f}% | Time: {res['train_time']:7.1f}s")
    print("="*50)
    
    # Write report
    print("\nWriting reports/classification_reports.txt...")
    with open("reports/classification_reports.txt", "w") as f:
        f.write("MODEL COMPARISON AND DETAILED CLASSIFICATION REPORTS\n")
        f.write("="*60 + "\n\n")
        
        for name in ['Improved CNN', 'LeNet-5', 'Mini-ResNet', 'SVM']:
            res = results[name]
            f.write(f"Model: {name}\n")
            f.write(f"MNIST Validation Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"USPS Test Accuracy (Cross-Domain): {res['usps_accuracy']:.4f}\n")
            f.write(f"Training Time: {res['train_time']:.2f} seconds\n")
            f.write("-"*40 + "\n")
            f.write(classification_report(y_val, res['preds']))
            f.write("\n" + "="*60 + "\n\n")
            
    # Generate Plots
    print("Generating reports/model_comparison.png...")
    plt.figure(figsize=(10, 5))
    model_names = ['Improved CNN', 'LeNet-5', 'Mini-ResNet', 'SVM']
    mnist_accs = [results[name]['accuracy'] * 100 for name in model_names]
    
    plt.subplot(1, 2, 1)
    sns.barplot(x=model_names, y=mnist_accs, palette='viridis')
    plt.title('Validation Accuracy (%)')
    plt.ylabel('%')
    plt.ylim(90, 100)
    plt.xticks(rotation=15)
    for i, v in enumerate(mnist_accs):
        plt.text(i, v + 0.2, f"{v:.2f}%", ha='center', fontweight='bold')
        
    plt.subplot(1, 2, 2)
    times = [results[name]['train_time'] for name in model_names]
    sns.barplot(x=model_names, y=times, palette='rocket')
    plt.title('Training Time (seconds)')
    plt.ylabel('Seconds')
    plt.xticks(rotation=15)
    for i, v in enumerate(times):
        plt.text(i, v + (max(times)*0.01), f"{v:.1f}s", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("reports/model_comparison.png")
    plt.close()
    
    print("Generating reports/mnist_vs_usps_comparison.png...")
    plt.figure(figsize=(10, 6))
    usps_accs = [results[name]['usps_accuracy'] * 100 for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, mnist_accs, width, label='MNIST Val Accuracy (In-Domain)', color='#4a69bd')
    plt.bar(x + width/2, usps_accs, width, label='USPS Test Accuracy (Cross-Domain)', color='#e55039')
    
    plt.ylabel('Accuracy (%)')
    plt.title('MNIST vs. USPS: Generalization Under Domain Shift')
    plt.xticks(x, model_names)
    plt.ylim(0, 115)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    for idx, val in enumerate(mnist_accs):
        plt.text(idx - width/2, val + 1.5, f"{val:.1f}%", ha='center', fontsize=9, fontweight='bold', color='#1e3799')
    for idx, val in enumerate(usps_accs):
        plt.text(idx + width/2, val + 1.5, f"{val:.1f}%", ha='center', fontsize=9, fontweight='bold', color='#b71540')
        
    plt.tight_layout()
    plt.savefig("reports/mnist_vs_usps_comparison.png")
    plt.close()
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
