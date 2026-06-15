import tensorflow as tf
from tensorflow.keras import layers, models

def build_improved_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds a modern CNN architecture with Batch Normalization and Dropout.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Fully Connected Block
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="improved_cnn")
    return model

def build_lenet5(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds the classic LeNet-5 architecture (using tanh & average pooling).
    """
    inputs = layers.Input(shape=input_shape)
    
    # Layer 1: Conv2D + AveragePooling
    x = layers.Conv2D(6, (5, 5), padding='same', activation='tanh')(inputs)
    x = layers.AveragePooling2D((2, 2))(x)
    
    # Layer 2: Conv2D + AveragePooling
    x = layers.Conv2D(16, (5, 5), padding='valid', activation='tanh')(x)
    x = layers.AveragePooling2D((2, 2))(x)
    
    # Layer 3: Conv2D (flattening conv)
    x = layers.Conv2D(120, (5, 5), padding='valid', activation='tanh')(x)
    
    # Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(84, activation='tanh')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="lenet5")
    return model

def residual_block(input_tensor, filters):
    """
    A helper function to create a standard residual block.
    """
    # First convolution layer
    x = layers.Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second convolution layer
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection (identity map since dims match)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def build_mini_resnet(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds a custom lightweight ResNet architecture optimized for 28x28 images.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Entry flow
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual block 1
    x = residual_block(x, 32)
    
    # Transition block (Downsample spatial dims, double channels)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual block 2
    x = residual_block(x, 64)
    
    # Exit flow
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="mini_resnet")
    return model

if __name__ == "__main__":
    # Print summaries of all models
    cnn = build_improved_cnn()
    cnn.summary()
    
    lenet = build_lenet5()
    lenet.summary()
    
    resnet = build_mini_resnet()
    resnet.summary()
