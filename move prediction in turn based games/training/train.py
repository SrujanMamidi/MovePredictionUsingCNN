import tensorflow as tf
import numpy as np
from models.resnet import build_chess_resnet
from training.data_loader import load_and_preprocess_data, get_label_encoder
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
CSV_PATH = 'assets/games.csv'
MODEL_SAVE_PATH = 'models/best_model.h5'
LOG_DIR = 'logs'
SAMPLE_SIZE = 2000  # Decrease for testing, increase for real training
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

def train():
    # 1. Load Data
    print("Loading and preprocessing data...")
    X, y_labels, y_values = load_and_preprocess_data(CSV_PATH, sample_size=SAMPLE_SIZE)
    
    # 2. Encode Labels
    print("Encoding labels...")
    encoder = get_label_encoder(y_labels)
    y_policy = encoder.transform(y_labels)
    num_classes = len(encoder.classes_)
    
    # 3. Split Data
    X_train, X_val, y_policy_train, y_policy_val, y_value_train, y_value_val = train_test_split(
        X, y_policy, y_values, test_size=0.1, random_state=42
    )
    
    # 4. Build Model
    print(f"Building ResNet model with {num_classes} output moves...")
    model = build_chess_resnet(input_shape=(8, 8, 17), num_moves=num_classes)
    
    # 5. Compile Model with AdamW and Top-K metrics
    # Note: Keras 3.0+ has AdamW. For older versions, we use Adam with weight decay.
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss={
            'policy': 'sparse_categorical_crossentropy',
            'value': 'mean_squared_error'
        },
        loss_weights={
            'policy': 1.0,
            'value': 0.5
        },
        metrics={
            'policy': ['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_acc')],
            'value': 'mae'
        }
    )
    
    # 6. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_policy_accuracy'),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
    ]
    
    # 7. Train
    print("Starting training...")
    history = model.fit(
        X_train, 
        {'policy': y_policy_train, 'value': y_value_train},
        validation_data=(X_val, {'policy': y_policy_val, 'value': y_value_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")
    return history

if __name__ == "__main__":
    train()
