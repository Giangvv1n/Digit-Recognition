import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.utils import to_categorical

# Import data loader and models
from data_loader import load_data, load_usps
from models import build_improved_cnn, build_lenet5, build_mini_resnet

def configure_data_augmentation():
    """
    Sets up a simple data augmentation pipeline.
    """
    return tf.keras.Sequential([
        layers.RandomRotation(factor=0.08),  # up to ~30 degrees
        layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
        layers.RandomZoom(height_factor=0.08, width_factor=0.08)
    ])

def prepare_tf_dataset(X, y, batch_size=128, augment=False, augmentation_layer=None):
    """
    Converts numpy arrays to tf.data.Dataset.
    Applies data augmentation if augment is True.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        ds = ds.shuffle(buffer_size=1024)
        ds = ds.batch(batch_size)
        if augmentation_layer is not None:
            ds = ds.map(lambda x, y: (augmentation_layer(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size)
    
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """
    Generates and saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load Data
    X_train, X_val, y_train, y_val, X_test = load_data(data_dir="data")
    
    # Load USPS data for out-of-domain evaluation
    print("\nLoading USPS dataset for cross-domain evaluation...")
    try:
        X_usps_train, y_usps_train, X_usps_test, y_usps_test = load_usps(data_dir="data")
        usps_ds_test = prepare_tf_dataset(X_usps_test, y_usps_test, batch_size=128, augment=False)
        usps_loaded = True
    except Exception as e:
        print(f"Warning: Could not load USPS dataset: {e}")
        usps_loaded = False
        
    # Pre-configure augmentation and dataset pipelines
    augment_layer = configure_data_augmentation()
    
    train_ds_aug = prepare_tf_dataset(X_train, y_train, batch_size=128, augment=True, augmentation_layer=augment_layer)
    train_ds_no_aug = prepare_tf_dataset(X_train, y_train, batch_size=128, augment=False)
    val_ds = prepare_tf_dataset(X_val, y_val, batch_size=128, augment=False)
    
    # Dictionary to keep track of model validation performances and train times
    results = {}
    histories = {}
    predictions = {}
    
    # Helper function to create fresh callbacks for each training run to avoid state sharing
    def create_callbacks():
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=2, 
            min_lr=1e-5, 
            verbose=1
        )
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True, 
            verbose=1
        )
        return [lr_scheduler, early_stop]
    
    # ==========================================
    # Model 1: Improved CNN (With Augmentation)
    # ==========================================
    print("\n" + "="*50)
    print("Training Model 1: Improved CNN (with BatchNorm & Dropout)")
    print("="*50)
    
    cnn_model = build_improved_cnn()
    cnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start_time = time.time()
    # Train with augmented data
    history_cnn = cnn_model.fit(
        train_ds_aug,
        epochs=12,  # Quick training (12 epochs is enough to get >99% on MNIST)
        validation_data=val_ds,
        callbacks=create_callbacks()
    )
    cnn_time = time.time() - start_time
    
    # Evaluate on MNIST validation
    val_loss, val_acc = cnn_model.evaluate(val_ds, verbose=0)
    print(f"Improved CNN Validation Accuracy: {val_acc:.4f}")
    
    # Evaluate on USPS test (out-of-domain)
    if usps_loaded:
        _, usps_acc = cnn_model.evaluate(usps_ds_test, verbose=0)
        print(f"Improved CNN USPS Accuracy: {usps_acc:.4f}")
    else:
        usps_acc = 0.0
    
    # Save model
    cnn_model.save("models/improved_cnn.keras")
    results['Improved CNN'] = {'accuracy': val_acc, 'usps_accuracy': usps_acc, 'train_time': cnn_time}
    histories['Improved CNN'] = history_cnn.history
    
    # Generate predictions
    cnn_preds = np.argmax(cnn_model.predict(val_ds), axis=1)
    predictions['Improved CNN'] = cnn_preds
    plot_confusion_matrix(y_val, cnn_preds, "Improved CNN Confusion Matrix", "reports/confusion_matrix_cnn.png")
    
    # ==========================================
    # Model 2: LeNet-5 (No Augmentation - Baseline)
    # ==========================================
    print("\n" + "="*50)
    print("Training Model 2: LeNet-5 (Classic)")
    print("="*50)
    
    lenet_model = build_lenet5()
    lenet_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start_time = time.time()
    # Train on standard data without augmentation
    history_lenet = lenet_model.fit(
        train_ds_no_aug,
        epochs=10,
        validation_data=val_ds,
        callbacks=create_callbacks()
    )
    lenet_time = time.time() - start_time
    
    # Evaluate on MNIST validation
    val_loss_len, val_acc_len = lenet_model.evaluate(val_ds, verbose=0)
    print(f"LeNet-5 Validation Accuracy: {val_acc_len:.4f}")
    
    # Evaluate on USPS test (out-of-domain)
    if usps_loaded:
        _, usps_acc_len = lenet_model.evaluate(usps_ds_test, verbose=0)
        print(f"LeNet-5 USPS Accuracy: {usps_acc_len:.4f}")
    else:
        usps_acc_len = 0.0
        
    # Save model
    lenet_model.save("models/lenet5.keras")
    results['LeNet-5'] = {'accuracy': val_acc_len, 'usps_accuracy': usps_acc_len, 'train_time': lenet_time}
    histories['LeNet-5'] = history_lenet.history
    
    # Generate predictions
    lenet_preds = np.argmax(lenet_model.predict(val_ds), axis=1)
    predictions['LeNet-5'] = lenet_preds
    plot_confusion_matrix(y_val, lenet_preds, "LeNet-5 Confusion Matrix", "reports/confusion_matrix_lenet.png")
    
    # ==========================================
    # Model 3: Mini-ResNet (With Augmentation)
    # ==========================================
    print("\n" + "="*50)
    print("Training Model 3: Mini-ResNet")
    print("="*50)
    
    resnet_model = build_mini_resnet()
    resnet_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start_time = time.time()
    # Train with augmented data
    history_resnet = resnet_model.fit(
        train_ds_aug,
        epochs=12,
        validation_data=val_ds,
        callbacks=create_callbacks()
    )
    resnet_time = time.time() - start_time
    
    # Evaluate on MNIST validation
    val_loss_res, val_acc_res = resnet_model.evaluate(val_ds, verbose=0)
    print(f"Mini-ResNet Validation Accuracy: {val_acc_res:.4f}")
    
    # Evaluate on USPS test (out-of-domain)
    if usps_loaded:
        _, usps_acc_res = resnet_model.evaluate(usps_ds_test, verbose=0)
        print(f"Mini-ResNet USPS Accuracy: {usps_acc_res:.4f}")
    else:
        usps_acc_res = 0.0
        
    # Save model
    resnet_model.save("models/mini_resnet.keras")
    results['Mini-ResNet'] = {'accuracy': val_acc_res, 'usps_accuracy': usps_acc_res, 'train_time': resnet_time}
    histories['Mini-ResNet'] = history_resnet.history
    
    # Generate predictions
    resnet_preds = np.argmax(resnet_model.predict(val_ds), axis=1)
    predictions['Mini-ResNet'] = resnet_preds
    plot_confusion_matrix(y_val, resnet_preds, "Mini-ResNet Confusion Matrix", "reports/confusion_matrix_resnet.png")
    
    # ==========================================
    # Model 4: Support Vector Machine (SVM)
    # ==========================================
    print("\n" + "="*50)
    print("Training Model 4: Support Vector Machine (with PCA)")
    print("="*50)
    
    # Flatten inputs for SVM
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    # If the dataset is large (e.g. 54,000 samples from official MNIST), training SVM can be slow.
    # We will sample a subset of 15,000 instances for SVM training to speed it up.
    # It still achieves excellent (~98%) accuracy.
    max_svm_samples = 15000
    if len(X_train_flat) > max_svm_samples:
        print(f"Subsampling SVM training data to {max_svm_samples} random samples for faster training...")
        idx = np.random.choice(len(X_train_flat), max_svm_samples, replace=False)
        X_train_svm_in = X_train_flat[idx]
        y_train_svm_in = y_train[idx]
    else:
        X_train_svm_in = X_train_flat
        y_train_svm_in = y_train
        
    start_time = time.time()
    
    # Dimensionality Reduction using PCA
    pca = PCA(n_components=50, random_state=42)
    X_train_pca = pca.fit_transform(X_train_svm_in)
    X_val_pca = pca.transform(X_val_flat)
    
    # Train SVM Classifier
    svm_clf = SVC(C=10.0, kernel='rbf', probability=True, random_state=42)
    svm_clf.fit(X_train_pca, y_train_svm_in)
    
    svm_time = time.time() - start_time
    
    # Evaluate
    svm_preds = svm_clf.predict(X_val_pca)
    svm_acc = accuracy_score(y_val, svm_preds)
    print(f"SVM Validation Accuracy: {svm_acc:.4f}")
    
    # Evaluate on USPS test (out-of-domain)
    if usps_loaded:
        X_usps_test_flat = X_usps_test.reshape(X_usps_test.shape[0], -1)
        X_usps_test_pca = pca.transform(X_usps_test_flat)
        usps_svm_preds = svm_clf.predict(X_usps_test_pca)
        usps_svm_acc = accuracy_score(y_usps_test, usps_svm_preds)
        print(f"SVM USPS Accuracy: {usps_svm_acc:.4f}")
    else:
        usps_svm_acc = 0.0
        
    # Save SVM & PCA models
    with open("models/svm_model.pkl", "wb") as f:
        pickle.dump(svm_clf, f)
    with open("models/pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
        
    results['SVM'] = {'accuracy': svm_acc, 'usps_accuracy': usps_svm_acc, 'train_time': svm_time}
    predictions['SVM'] = svm_preds
    plot_confusion_matrix(y_val, svm_preds, "SVM Confusion Matrix", "reports/confusion_matrix_svm.png")
    
    # ==========================================
    # Post-Training Comparisons and Visualization
    # ==========================================
    print("\n" + "="*50)
    print("Generating Comparative Reports and Plots...")
    print("="*50)
    
    # 1. Plot Training Histories (DL Models)
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    for model_name, hist in histories.items():
        plt.plot(hist['accuracy'], label=f'{model_name} Train')
        plt.plot(hist['val_accuracy'], linestyle='--', label=f'{model_name} Val')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    for model_name, hist in histories.items():
        plt.plot(hist['loss'], label=f'{model_name} Train')
        plt.plot(hist['val_loss'], linestyle='--', label=f'{model_name} Val')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("reports/training_histories.png")
    plt.close()
    
    # 2. Print Classification Report for each model
    with open("reports/classification_reports.txt", "w") as f:
        f.write("MODEL COMPARISON AND DETAILED CLASSIFICATION REPORTS\n")
        f.write("="*60 + "\n\n")
        
        for name, res in results.items():
            f.write(f"Model: {name}\n")
            f.write(f"MNIST Validation Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"USPS Test Accuracy (Cross-Domain): {res['usps_accuracy']:.4f}\n")
            f.write(f"Training Time: {res['train_time']:.2f} seconds\n")
            f.write("-"*40 + "\n")
            f.write(classification_report(y_val, predictions[name]))
            f.write("\n" + "="*60 + "\n\n")
            
    # 3. Bar Chart Comparison: Accuracy and Time
    plt.figure(figsize=(10, 5))
    
    # Accuracy bars
    plt.subplot(1, 2, 1)
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in model_names]
    sns.barplot(x=model_names, y=accuracies, palette='viridis')
    plt.title('Validation Accuracy (%)')
    plt.ylabel('%')
    plt.ylim(90, 100)
    plt.xticks(rotation=15)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.2, f"{v:.2f}%", ha='center', fontweight='bold')
        
    # Time bars
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
    # 4. Generate MNIST vs USPS Side-by-Side Plot (Generalization comparison)
    if usps_loaded:
        plt.figure(figsize=(10, 6))
        model_names = list(results.keys())
        mnist_accs = [results[name]['accuracy'] * 100 for name in model_names]
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
        
        # Add values on top of bars
        for idx, val in enumerate(mnist_accs):
            plt.text(idx - width/2, val + 1.5, f"{val:.1f}%", ha='center', fontsize=9, fontweight='bold', color='#1e3799')
        for idx, val in enumerate(usps_accs):
            plt.text(idx + width/2, val + 1.5, f"{val:.1f}%", ha='center', fontsize=9, fontweight='bold', color='#b71540')
            
        plt.tight_layout()
        plt.savefig("reports/mnist_vs_usps_comparison.png")
        plt.close()
        print("Generated MNIST vs USPS Generalization Plot at: reports/mnist_vs_usps_comparison.png")

    print("\nTraining and comparison completed successfully!")
    print("Trained models saved in: models/")
    print("Reports and charts generated in: reports/")

if __name__ == "__main__":
    main()
