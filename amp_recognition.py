"""
Deep Learning Model for Antimicrobial Peptide (AMP) Classification
Uses CNN-LSTM hybrid architecture to classify peptide sequences as AMP or non-AMP
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Amino acid vocabulary (20 standard amino acids)
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}  # 0 reserved for padding
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}
VOCAB_SIZE = len(AMINO_ACIDS) + 1  # +1 for padding token


def parse_fasta(file_path: str) -> List[str]:
    """
    Parse FASTA file and extract sequences.
    
    Args:
        file_path: Path to FASTA file
        
    Returns:
        List of amino acid sequences
    """
    sequences = []
    current_seq = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                if line:
                    current_seq.append(line.upper())
        
        # Add last sequence
        if current_seq:
            sequences.append(''.join(current_seq))
    
    return sequences


def load_dataset(data_dir: str = 'dataset') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training, validation, and test datasets.
    
    Args:
        data_dir: Directory containing FASTA files
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Load AMP sequences
    amp_train = parse_fasta(os.path.join(data_dir, 'AMP.tr.fa'))
    amp_eval = parse_fasta(os.path.join(data_dir, 'AMP.eval.fa'))
    amp_test = parse_fasta(os.path.join(data_dir, 'AMP.te.fa'))
    
    # Load DECOY (non-AMP) sequences
    decoy_train = parse_fasta(os.path.join(data_dir, 'DECOY.tr.fa'))
    decoy_eval = parse_fasta(os.path.join(data_dir, 'DECOY.eval.fa'))
    decoy_test = parse_fasta(os.path.join(data_dir, 'DECOY.te.fa'))
    
    # Combine and create labels
    X_train = amp_train + decoy_train
    y_train = [1] * len(amp_train) + [0] * len(decoy_train)
    
    X_val = amp_eval + decoy_eval
    y_val = [1] * len(amp_eval) + [0] * len(decoy_eval)
    
    X_test = amp_test + decoy_test
    y_test = [1] * len(amp_test) + [0] * len(decoy_test)
    
    # Shuffle training data
    train_indices = np.random.permutation(len(X_train))
    X_train = [X_train[i] for i in train_indices]
    y_train = [y_train[i] for i in train_indices]
    
    print(f"Training set: {len(X_train)} sequences ({sum(y_train)} AMP, {len(y_train) - sum(y_train)} non-AMP)")
    print(f"Validation set: {len(X_val)} sequences ({sum(y_val)} AMP, {len(y_val) - sum(y_val)} non-AMP)")
    print(f"Test set: {len(X_test)} sequences ({sum(y_test)} AMP, {len(y_test) - sum(y_test)} non-AMP)")
    
    return X_train, np.array(y_train), X_val, np.array(y_val), X_test, np.array(y_test)


def encode_sequences(sequences: List[str], max_length: int) -> np.ndarray:
    """
    Encode amino acid sequences to integer indices.
    
    Args:
        sequences: List of amino acid sequences
        max_length: Maximum sequence length (sequences will be padded/truncated)
        
    Returns:
        Numpy array of shape (n_sequences, max_length)
    """
    encoded = np.zeros((len(sequences), max_length), dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_length]):
            if aa in AA_TO_IDX:
                encoded[i, j] = AA_TO_IDX[aa]
            # Unknown amino acids remain 0 (padding)
    
    return encoded


def analyze_sequence_lengths(sequences: List[str]) -> dict:
    """
    Analyze sequence length distribution.
    
    Args:
        sequences: List of sequences
        
    Returns:
        Dictionary with statistics
    """
    lengths = [len(seq) for seq in sequences]
    return {
        'min': min(lengths),
        'max': max(lengths),
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'percentile_95': np.percentile(lengths, 95),
        'percentile_99': np.percentile(lengths, 99)
    }


def build_model(max_length: int, vocab_size: int, embedding_dim: int = 128) -> keras.Model:
    """
    Build CNN-LSTM hybrid model for AMP classification.
    
    Args:
        max_length: Maximum sequence length
        vocab_size: Vocabulary size (number of amino acids + 1 for padding)
        embedding_dim: Embedding dimension
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(max_length,), name='sequence_input')
    
    # Embedding layer
    x = layers.Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True)(inputs)
    
    # CNN block for local pattern extraction
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Bidirectional LSTM for long-range dependencies
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(32, dropout=0.3))(x)
    
    # Classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='amp_prediction')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='AMP_Classifier')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), 
                 keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
    )
    
    return model


def train_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, 
                batch_size: int = 32, epochs: int = 50,
                model_save_path: str = 'amp_model.keras') -> keras.callbacks.History:
    """
    Train the model with early stopping and checkpointing.
    
    Args:
        model: Compiled Keras model
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        model_save_path: Path to save best model
        
    Returns:
        Training history
    """
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    return history


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model on test set with comprehensive metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test sequences
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics


def plot_training_history(history: keras.callbacks.History, save_path: str = 'training_history.png'):
    """
    Plot training curves.
    
    Args:
        history: Training history from model.fit()
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # AUC
    axes[1, 1].plot(history.history['auc'], label='Training AUC')
    axes[1, 1].plot(history.history['val_auc'], label='Validation AUC')
    axes[1, 1].set_title('Model AUC-ROC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: str = 'confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Non-AMP', 'AMP'],
           yticklabels=['Non-AMP', 'AMP'],
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to {save_path}")
    plt.close()


def predict_sequence(model: keras.Model, sequence: str, max_length: int) -> Tuple[float, str]:
    """
    Predict if a single sequence is an AMP.
    
    Args:
        model: Trained Keras model
        sequence: Amino acid sequence
        max_length: Maximum sequence length used during training
        
    Returns:
        Tuple of (probability, prediction label)
    """
    # Encode sequence
    encoded = encode_sequences([sequence.upper()], max_length)
    
    # Predict
    proba = model.predict(encoded, verbose=0)[0][0]
    label = 'AMP' if proba > 0.5 else 'Non-AMP'
    
    return proba, label


def main():
    """Main training and evaluation pipeline."""
    print("=" * 60)
    print("AMP Classification Model - Training Pipeline")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    
    # Analyze sequence lengths
    print("\n[2/6] Analyzing sequence lengths...")
    all_sequences = X_train + X_val + X_test
    length_stats = analyze_sequence_lengths(all_sequences)
    print(f"Sequence length statistics:")
    print(f"  Min: {length_stats['min']}")
    print(f"  Max: {length_stats['max']}")
    print(f"  Mean: {length_stats['mean']:.2f}")
    print(f"  Median: {length_stats['median']:.2f}")
    print(f"  95th percentile: {length_stats['percentile_95']:.2f}")
    print(f"  99th percentile: {length_stats['percentile_99']:.2f}")
    
    # Set max_length (use 99th percentile to capture most sequences)
    max_length = int(length_stats['percentile_99'])
    print(f"\nUsing max_length = {max_length}")
    
    # Encode sequences
    print("\n[3/6] Encoding sequences...")
    X_train_encoded = encode_sequences(X_train, max_length)
    X_val_encoded = encode_sequences(X_val, max_length)
    X_test_encoded = encode_sequences(X_test, max_length)
    
    # Build model
    print("\n[4/6] Building model...")
    model = build_model(max_length, VOCAB_SIZE)
    print(model.summary())
    
    # Train model
    print("\n[5/6] Training model...")
    history = train_model(
        model, X_train_encoded, y_train,
        X_val_encoded, y_val,
        batch_size=32,
        epochs=50,
        model_save_path='amp_model.keras'
    )
    
    # Evaluate model
    print("\n[6/6] Evaluating model on test set...")
    metrics = evaluate_model(model, X_test_encoded, y_test)
    
    print("\n" + "=" * 60)
    print("Test Set Performance Metrics")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Non-AMP   AMP")
    print(f"True Non-AMP    {metrics['confusion_matrix'][0,0]:4d}   {metrics['confusion_matrix'][0,1]:4d}")
    print(f"True AMP        {metrics['confusion_matrix'][1,0]:4d}   {metrics['confusion_matrix'][1,1]:4d}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_training_history(history, 'training_history.png')
    plot_confusion_matrix(metrics['confusion_matrix'], 'confusion_matrix.png')
    
    # Save preprocessing parameters
    import json
    preprocess_params = {
        'max_length': max_length,
        'vocab_size': VOCAB_SIZE,
        'amino_acids': AMINO_ACIDS
    }
    with open('preprocess_params.json', 'w') as f:
        json.dump(preprocess_params, f, indent=2)
    print("Preprocessing parameters saved to preprocess_params.json")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Model saved to: amp_model.keras")
    print(f"Use predict_sequence() function for inference on new sequences.")


if __name__ == "__main__":
    main()
