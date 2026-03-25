import tensorflow as tf
from tensorflow.keras import layers, models

def resnet_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_chess_resnet(input_shape=(8, 8, 17), num_moves=2000, num_res_blocks=10):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Convolution
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Residual Blocks
    for _ in range(num_res_blocks):
        x = resnet_block(x, 64)
        
    # --- Policy Head ---
    policy_x = layers.Conv2D(2, 1, padding='same')(x)
    policy_x = layers.BatchNormalization()(policy_x)
    policy_x = layers.ReLU()(policy_x)
    policy_x = layers.Flatten()(policy_x)
    policy_output = layers.Dense(num_moves, activation='softmax', name='policy')(policy_x)
    
    # --- Value Head ---
    value_x = layers.Conv2D(1, 1, padding='same')(x)
    value_x = layers.BatchNormalization()(value_x)
    value_x = layers.ReLU()(value_x)
    value_x = layers.Flatten()(value_x)
    value_x = layers.Dense(64, activation='relu')(value_x)
    value_x = layers.Dropout(0.3)(value_x)
    value_output = layers.Dense(1, activation='tanh', name='value')(value_x)
    
    model = models.Model(inputs=inputs, outputs=[policy_output, value_output])
    return model
