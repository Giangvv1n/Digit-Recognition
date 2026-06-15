import os
import gzip
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

def load_data(data_dir="data", val_size=0.1, random_state=42):
    """
    Loads train and test data for MNIST.
    If Kaggle CSV files are found in data_dir/train.csv and data_dir/test.csv, it loads them.
    Otherwise, it falls back to tensorflow.keras.datasets.mnist.
    
    Returns:
        X_train, X_val, y_train, y_val, X_test (all normalized)
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if os.path.exists(train_path):
        print(f"Loading local train dataset from {train_path}...")
        train_df = pd.read_csv(train_path)
        
        # Split labels and pixels
        y = train_df.iloc[:, 0].values
        X = train_df.iloc[:, 1:].values
        
        # Reshape to (N, 28, 28, 1) and normalize to [0, 1]
        X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state, stratify=y
        )
        
        # Load test set if it exists
        if os.path.exists(test_path):
            print(f"Loading local test dataset from {test_path}...")
            test_df = pd.read_csv(test_path)
            X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        else:
            print("Local test.csv not found. Creating a test set from validation data.")
            # Use validation data as mock test if no test set exists
            X_test = X_val
            
    else:
        mnist_local_path = os.path.join(data_dir, "mnist.npz")
        if os.path.exists(mnist_local_path):
            print(f"Loading local MNIST dataset from cache: {mnist_local_path}...")
            cached_data = np.load(mnist_local_path)
            X_train_raw = cached_data['X_train']
            y_train_raw = cached_data['y_train']
            X_test_raw = cached_data['X_test']
            y_test_raw = cached_data['y_test']
        else:
            print("Local mnist.npz not found. Downloading official Keras MNIST dataset...")
            (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()
            print(f"Saving MNIST dataset locally to {mnist_local_path}...")
            np.savez_compressed(
                mnist_local_path, 
                X_train=X_train_raw, 
                y_train=y_train_raw, 
                X_test=X_test_raw, 
                y_test=y_test_raw
            )
        
        # Add channel dimension
        X_train_all = X_train_raw.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        X_test = X_test_raw.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_train_all = y_train_raw
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all, y_train_all, test_size=val_size, random_state=random_state, stratify=y_train_all
        )
        
    print(f"Dataset summary:")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    
    return X_train, X_val, y_train, y_val, X_test

def load_usps(data_dir="data"):
    """
    Downloads and loads the USPS digit dataset using sklearn.datasets.fetch_openml.
    Saves it locally to data/usps.npz so the user has physical files in the folder.
    Resizes the 16x16 images to 28x28 grayscale so they match MNIST formatting.
    """
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, "usps.npz")
    
    # If already cached locally, load from file directly
    if os.path.exists(local_path):
        print(f"Loading local USPS dataset from cache: {local_path}...")
        cached_data = np.load(local_path)
        X_train = cached_data['X_train']
        y_train = cached_data['y_train']
        X_test = cached_data['X_test']
        y_test = cached_data['y_test']
        
        print(f"USPS Dataset loaded successfully from local file:")
        print(f"  USPS Train shape: {X_train.shape}, Labels: {y_train.shape}")
        print(f"  USPS Test shape:  {X_test.shape}, Labels: {y_test.shape}")
        return X_train, y_train, X_test, y_test

    print("Loading USPS dataset via sklearn fetch_openml...")
    from sklearn.datasets import fetch_openml
    usps = fetch_openml(name='usps', version=2, as_frame=False)
    
    # Extract data and labels
    X_all = usps.data  # shape (9298, 256), values between -1.0 and 1.0
    y_all = usps.target.astype(int) - 1  # class 1 -> digit 0, ..., class 10 -> digit 9
    
    # Original USPS split: first 7291 are train, remaining 2007 are test
    X_train_raw = X_all[:7291]
    y_train = y_all[:7291]
    X_test_raw = X_all[7291:]
    y_test = y_all[7291:]
    
    # Preprocess: rescale from [-1, 1] to [0, 1]
    X_train_raw = (X_train_raw + 1.0) / 2.0
    X_test_raw = (X_test_raw + 1.0) / 2.0
    
    # Reshape to 16x16 images
    X_train_16 = X_train_raw.reshape(-1, 16, 16)
    X_test_16 = X_test_raw.reshape(-1, 16, 16)
    
    # Resize to 28x28 using PIL
    X_train_28 = []
    for img in X_train_16:
        pil_img = Image.fromarray((img * 255.0).astype('uint8'))
        resized = pil_img.resize((28, 28), Image.Resampling.BILINEAR)
        X_train_28.append(np.array(resized).astype('float32') / 255.0)
        
    X_test_28 = []
    for img in X_test_16:
        pil_img = Image.fromarray((img * 255.0).astype('uint8'))
        resized = pil_img.resize((28, 28), Image.Resampling.BILINEAR)
        X_test_28.append(np.array(resized).astype('float32') / 255.0)
        
    X_train = np.expand_dims(np.array(X_train_28), axis=-1)
    X_test = np.expand_dims(np.array(X_test_28), axis=-1)
    
    # Save locally to npz file so user has physical files
    print(f"Saving USPS dataset locally to {local_path}...")
    np.savez_compressed(local_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
    print(f"USPS Dataset loaded successfully:")
    print(f"  USPS Train shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"  USPS Test shape:  {X_test.shape}, Labels: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Test loader behavior
    os.makedirs("data", exist_ok=True)
    load_data(data_dir="data")
    load_usps(data_dir="data")
