import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import keras_tuner as kt

# Import data loader
from data_loader import load_data

def build_tunable_model(hp):
    """
    Builds a CNN model with search spaces for Keras Tuner.
    """
    inputs = layers.Input(shape=(28, 28, 1))
    
    # Tune number of filters in the first conv block
    filters_1 = hp.Int('conv_1_filter', min_value=16, max_value=48, step=16)
    x = layers.Conv2D(filters_1, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters_1, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    dropout_1 = hp.Float('dropout_1', min_value=0.2, max_value=0.4, step=0.1)
    x = layers.Dropout(dropout_1)(x)
    
    # Tune number of filters in the second conv block
    filters_2 = hp.Int('conv_2_filter', min_value=32, max_value=96, step=32)
    x = layers.Conv2D(filters_2, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters_2, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    dropout_2 = hp.Float('dropout_2', min_value=0.2, max_value=0.4, step=0.1)
    x = layers.Dropout(dropout_2)(x)
    
    # Fully connected block
    x = layers.Flatten()(x)
    dense_units = hp.Int('dense_units', min_value=128, max_value=256, step=64)
    x = layers.Dense(dense_units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    dropout_dense = hp.Float('dropout_dense', min_value=0.3, max_value=0.6, step=0.1)
    x = layers.Dropout(dropout_dense)(x)
    
    outputs = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Tune learning rate
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Load dataset
    X_train, X_val, y_train, y_val, _ = load_data(data_dir="data")
    
    # To speed up tuning, we sub-sample data if it is large
    # A smaller subset is perfect for finding good hyperparameters quickly
    max_tune_samples = 10000
    if len(X_train) > max_tune_samples:
        print(f"Subsampling tuning dataset to {max_tune_samples} samples for fast search...")
        X_train = X_train[:max_tune_samples]
        y_train = y_train[:max_tune_samples]
        X_val = X_val[:2000]
        y_val = y_val[:2000]
        
    print("Initializing Keras RandomSearch Tuner...")
    tuner = kt.RandomSearch(
        build_tunable_model,
        objective='val_accuracy',
        max_trials=3,          # Small search space for quick execution
        executions_per_trial=1,
        directory='tuning_results',
        project_name='digit_recognizer_tuning',
        overwrite=True
    )
    
    # Run the tuning process
    print("Starting hyperparameter tuning...")
    tuner.search(
        X_train, y_train,
        epochs=3,
        validation_data=(X_val, y_val),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)]
    )
    
    # Get results
    print("\nTuning completed! Summary of results:")
    tuner.results_summary()
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n" + "="*40)
    print("BEST HYPERPARAMETERS FOUND:")
    print("="*40)
    print(f"  Conv Block 1 Filters: {best_hps.get('conv_1_filter')}")
    print(f"  Conv Block 1 Dropout: {best_hps.get('dropout_1'):.2f}")
    print(f"  Conv Block 2 Filters: {best_hps.get('conv_2_filter')}")
    print(f"  Conv Block 2 Dropout: {best_hps.get('dropout_2'):.2f}")
    print(f"  Dense Units:          {best_hps.get('dense_units')}")
    print(f"  Dense Dropout:        {best_hps.get('dropout_dense'):.2f}")
    print(f"  Learning Rate:        {best_hps.get('learning_rate')}")
    print("="*40)

if __name__ == "__main__":
    main()
