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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from collections import Counter
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import BioPython for molecular weight calculations
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not available. Using simplified molecular weight calculation.")

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


# Amino acid properties for feature engineering
# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Amino acid charges (at pH 7)
CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

# Polarity index (Grantham, 1974)
POLARITY = {
    'A': 8.1, 'C': 5.5, 'D': 13.0, 'E': 12.5, 'F': 5.2,
    'G': 9.0, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9,
    'M': 5.7, 'N': 11.6, 'P': 8.0, 'Q': 10.5, 'R': 10.5,
    'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2
}

# Molecular weights (Da)
MOLECULAR_WEIGHT = {
    'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
    'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
}

# Amino acid groups
HYDROPHOBIC_AAS = set('AILMFWV')
POLAR_AAS = set('STNQ')
POSITIVE_CHARGED_AAS = set('KRH')
NEGATIVE_CHARGED_AAS = set('DE')
AROMATIC_AAS = set('FWY')
ALIPHATIC_AAS = set('AILV')


def extract_physicochemical_features(sequence: str) -> Dict[str, float]:
    """
    Extract physicochemical properties from a sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary of physicochemical features
    """
    seq = sequence.upper()
    length = len(seq)
    
    if length == 0:
        return {
            'hydrophobicity_avg': 0.0,
            'hydrophobicity_total': 0.0,
            'net_charge': 0.0,
            'positive_charge_count': 0.0,
            'negative_charge_count': 0.0,
            'charge_density': 0.0,
            'polarity_avg': 0.0,
            'aromaticity': 0.0,
            'molecular_weight': 0.0,
            'isoelectric_point_est': 7.0
        }
    
    # Hydrophobicity
    hydrophobicity_values = [HYDROPHOBICITY.get(aa, 0.0) for aa in seq]
    hydrophobicity_avg = np.mean(hydrophobicity_values)
    hydrophobicity_total = sum(hydrophobicity_values)
    
    # Charge
    charge_values = [CHARGE.get(aa, 0) for aa in seq]
    net_charge = sum(charge_values)
    positive_charge_count = sum(1 for c in charge_values if c > 0)
    negative_charge_count = sum(1 for c in charge_values if c < 0)
    charge_density = net_charge / length if length > 0 else 0.0
    
    # Polarity
    polarity_values = [POLARITY.get(aa, 0.0) for aa in seq]
    polarity_avg = np.mean(polarity_values)
    
    # Aromaticity (fraction of aromatic amino acids)
    aromatic_count = sum(1 for aa in seq if aa in AROMATIC_AAS)
    aromaticity = aromatic_count / length if length > 0 else 0.0
    
    # Molecular weight
    mw_values = [MOLECULAR_WEIGHT.get(aa, 0.0) for aa in seq]
    # Subtract water for each peptide bond (n-1 bonds for n residues)
    molecular_weight = sum(mw_values) - (length - 1) * 18.015
    
    # Simplified isoelectric point estimation
    # Based on charge balance: more positive = higher pI, more negative = lower pI
    if net_charge > 0:
        isoelectric_point_est = 7.0 + min(net_charge * 0.5, 7.0)  # Cap at ~14
    elif net_charge < 0:
        isoelectric_point_est = 7.0 + max(net_charge * 0.5, -7.0)  # Cap at ~0
    else:
        isoelectric_point_est = 7.0
    
    return {
        'hydrophobicity_avg': hydrophobicity_avg,
        'hydrophobicity_total': hydrophobicity_total,
        'net_charge': net_charge,
        'positive_charge_count': positive_charge_count,
        'negative_charge_count': negative_charge_count,
        'charge_density': charge_density,
        'polarity_avg': polarity_avg,
        'aromaticity': aromaticity,
        'molecular_weight': molecular_weight,
        'isoelectric_point_est': isoelectric_point_est
    }


def extract_composition_features(sequence: str) -> Dict[str, float]:
    """
    Extract amino acid composition features.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary of composition features
    """
    seq = sequence.upper()
    length = len(seq)
    
    if length == 0:
        # Return zeros for all composition features
        features = {aa: 0.0 for aa in AMINO_ACIDS}
        features.update({
            'hydrophobic_freq': 0.0,
            'polar_freq': 0.0,
            'positive_charged_freq': 0.0,
            'negative_charged_freq': 0.0,
            'aromatic_freq': 0.0,
            'aliphatic_freq': 0.0
        })
        return features
    
    # Count amino acids
    aa_counts = Counter(seq)
    
    # Individual amino acid frequencies
    features = {}
    for aa in AMINO_ACIDS:
        features[aa] = aa_counts.get(aa, 0) / length
    
    # Group frequencies
    hydrophobic_count = sum(1 for aa in seq if aa in HYDROPHOBIC_AAS)
    polar_count = sum(1 for aa in seq if aa in POLAR_AAS)
    positive_charged_count = sum(1 for aa in seq if aa in POSITIVE_CHARGED_AAS)
    negative_charged_count = sum(1 for aa in seq if aa in NEGATIVE_CHARGED_AAS)
    aromatic_count = sum(1 for aa in seq if aa in AROMATIC_AAS)
    aliphatic_count = sum(1 for aa in seq if aa in ALIPHATIC_AAS)
    
    features.update({
        'hydrophobic_freq': hydrophobic_count / length,
        'polar_freq': polar_count / length,
        'positive_charged_freq': positive_charged_count / length,
        'negative_charged_freq': negative_charged_count / length,
        'aromatic_freq': aromatic_count / length,
        'aliphatic_freq': aliphatic_count / length
    })
    
    return features


def extract_sequence_features(sequence: str) -> Dict[str, float]:
    """
    Extract sequence-based features.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary of sequence features
    """
    seq = sequence.upper()
    length = len(seq)
    
    if length == 0:
        return {
            'length': 0.0,
            'n_terminal_charge': 0.0,
            'c_terminal_charge': 0.0,
            'n_terminal_hydrophobicity': 0.0,
            'c_terminal_hydrophobicity': 0.0,
            'n_terminal_5_charge': 0.0,
            'c_terminal_5_charge': 0.0
        }
    
    # Length (will be normalized later)
    length_feature = float(length)
    
    # Terminal properties
    n_terminal_aa = seq[0] if seq else 'A'
    c_terminal_aa = seq[-1] if seq else 'A'
    n_terminal_charge = CHARGE.get(n_terminal_aa, 0)
    c_terminal_charge = CHARGE.get(c_terminal_aa, 0)
    n_terminal_hydrophobicity = HYDROPHOBICITY.get(n_terminal_aa, 0.0)
    c_terminal_hydrophobicity = HYDROPHOBICITY.get(c_terminal_aa, 0.0)
    
    # Charge at first/last 5 residues
    n_terminal_5 = seq[:5]
    c_terminal_5 = seq[-5:] if len(seq) >= 5 else seq
    n_terminal_5_charge = sum(CHARGE.get(aa, 0) for aa in n_terminal_5)
    c_terminal_5_charge = sum(CHARGE.get(aa, 0) for aa in c_terminal_5)
    
    return {
        'length': length_feature,
        'n_terminal_charge': float(n_terminal_charge),
        'c_terminal_charge': float(c_terminal_charge),
        'n_terminal_hydrophobicity': n_terminal_hydrophobicity,
        'c_terminal_hydrophobicity': c_terminal_hydrophobicity,
        'n_terminal_5_charge': float(n_terminal_5_charge),
        'c_terminal_5_charge': float(c_terminal_5_charge)
    }


def extract_all_features(sequences: List[str]) -> np.ndarray:
    """
    Extract all features for a list of sequences.
    
    Args:
        sequences: List of amino acid sequences
        
    Returns:
        Numpy array of shape (n_sequences, n_features)
    """
    all_feature_dicts = []
    
    for seq in sequences:
        # Extract all feature categories
        physico_features = extract_physicochemical_features(seq)
        comp_features = extract_composition_features(seq)
        seq_features = extract_sequence_features(seq)
        
        # Combine all features
        combined_features = {**physico_features, **comp_features, **seq_features}
        all_feature_dicts.append(combined_features)
    
    # Convert to array - need consistent feature order
    if len(all_feature_dicts) == 0:
        return np.array([])
    
    # Get feature names in consistent order
    feature_names = sorted(all_feature_dicts[0].keys())
    
    # Build array
    feature_array = np.zeros((len(sequences), len(feature_names)))
    for i, feat_dict in enumerate(all_feature_dicts):
        feature_array[i] = [feat_dict[name] for name in feature_names]
    
    return feature_array.astype(np.float32)


def normalize_features(X_features: np.ndarray, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.
    
    Args:
        X_features: Feature array of shape (n_samples, n_features)
        scaler: Optional pre-fitted scaler (for validation/test sets)
        
    Returns:
        Tuple of (normalized_features, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        X_features_normalized = scaler.fit_transform(X_features)
    else:
        X_features_normalized = scaler.transform(X_features)
    
    return X_features_normalized.astype(np.float32), scaler


def build_model(max_length: int, vocab_size: int, feature_dim: int, embedding_dim: int = 128) -> keras.Model:
    """
    Build CNN-LSTM hybrid model with multi-input (sequence + engineered features) for AMP classification.
    
    Args:
        max_length: Maximum sequence length
        vocab_size: Vocabulary size (number of amino acids + 1 for padding)
        feature_dim: Dimension of engineered features
        embedding_dim: Embedding dimension
        
    Returns:
        Compiled Keras model
    """
    # Sequence input branch
    sequence_input = layers.Input(shape=(max_length,), name='sequence_input')
    
    # Embedding layer
    x_seq = layers.Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True)(sequence_input)
    
    # CNN block for local pattern extraction
    x_seq = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x_seq)
    x_seq = layers.BatchNormalization()(x_seq)
    x_seq = layers.MaxPooling1D(pool_size=2)(x_seq)
    
    x_seq = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x_seq)
    x_seq = layers.BatchNormalization()(x_seq)
    x_seq = layers.MaxPooling1D(pool_size=2)(x_seq)
    
    # Bidirectional LSTM for long-range dependencies
    x_seq = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x_seq)
    x_seq = layers.Bidirectional(layers.LSTM(32, dropout=0.3))(x_seq)
    
    # Feature input branch
    feature_input = layers.Input(shape=(feature_dim,), name='feature_input')
    
    # Process features through dense layers
    x_feat = layers.Dense(64, activation='relu')(feature_input)
    x_feat = layers.BatchNormalization()(x_feat)
    x_feat = layers.Dropout(0.3)(x_feat)
    x_feat = layers.Dense(32, activation='relu')(x_feat)
    x_feat = layers.Dropout(0.2)(x_feat)
    
    # Concatenate sequence and feature branches
    x = layers.Concatenate()([x_seq, x_feat])
    
    # Classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='amp_prediction')(x)
    
    model = models.Model(inputs=[sequence_input, feature_input], outputs=outputs, name='AMP_Classifier')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), 
                 keras.metrics.Recall(name='sensitivity'), keras.metrics.AUC(name='auc')]
    )
    
    return model


def train_model(model: keras.Model, X_train_seq: np.ndarray, X_train_feat: np.ndarray, 
                y_train: np.ndarray, X_val_seq: np.ndarray, X_val_feat: np.ndarray,
                y_val: np.ndarray, batch_size: int = 32, epochs: int = 50,
                model_save_path: str = 'amp_model.keras') -> keras.callbacks.History:
    """
    Train the model with early stopping and checkpointing.
    
    Args:
        model: Compiled Keras model
        X_train_seq: Training sequences (encoded)
        X_train_feat: Training features
        y_train: Training labels
        X_val_seq: Validation sequences (encoded)
        X_val_feat: Validation features
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
    
    # Train model with multiple inputs
    history = model.fit(
        [X_train_seq, X_train_feat], y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_val_seq, X_val_feat], y_val),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    return history


def evaluate_model(model: keras.Model, X_test_seq: np.ndarray, X_test_feat: np.ndarray, 
                   y_test: np.ndarray) -> dict:
    """
    Evaluate model on test set with comprehensive metrics.
    
    Args:
        model: Trained Keras model
        X_test_seq: Test sequences (encoded)
        X_test_feat: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    # Predictions
    y_pred_proba = model.predict([X_test_seq, X_test_feat], verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    # Calculate Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    # Calculate Matthews Correlation Coefficient (MCC)
    denominator = np.sqrt((TP + FN) * (TN + FP) * (TP + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / denominator if denominator > 0 else 0.0
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'sensitivity': recall_score(y_test, y_pred),  # Sensitivity = Recall = True Positive Rate
        'f1_score': f1_score(y_test, y_pred),
        'specificity': specificity,
        'mcc': mcc,
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': cm
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


def load_feature_scaler(scaler_path: str = 'feature_scaler.pkl') -> Optional[StandardScaler]:
    """
    Load the feature scaler from file.
    
    Args:
        scaler_path: Path to the saved scaler file
        
    Returns:
        StandardScaler object or None if file doesn't exist
    """
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        print(f"Warning: Scaler file {scaler_path} not found. Features will be normalized without fitted scaler.")
        return None


def predict_sequence(model: keras.Model, sequence: str, max_length: int, 
                    scaler: Optional[StandardScaler] = None, 
                    scaler_path: str = 'feature_scaler.pkl') -> Tuple[float, str]:
    """
    Predict if a single sequence is an AMP.
    
    Args:
        model: Trained Keras model
        sequence: Amino acid sequence
        max_length: Maximum sequence length used during training
        scaler: Fitted StandardScaler for feature normalization (optional)
        scaler_path: Path to saved scaler file (used if scaler is None)
        
    Returns:
        Tuple of (probability, prediction label)
    """
    # Load scaler if not provided
    if scaler is None:
        scaler = load_feature_scaler(scaler_path)
    
    # Encode sequence
    encoded_seq = encode_sequences([sequence.upper()], max_length)
    
    # Extract and normalize features
    features = extract_all_features([sequence.upper()])
    if scaler is not None:
        features, _ = normalize_features(features, scaler)
    else:
        # If no scaler provided, use default normalization (shouldn't happen in practice)
        print("Warning: No scaler available. Using default normalization which may not match training.")
        features, _ = normalize_features(features)
    
    # Predict
    proba = model.predict([encoded_seq, features], verbose=0)[0][0]
    label = 'AMP' if proba > 0.5 else 'Non-AMP'
    
    return proba, label


def main():
    """Main training and evaluation pipeline."""
    print("=" * 60)
    print("AMP Classification Model - Training Pipeline (with Feature Engineering)")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/7] Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    
    # Analyze sequence lengths
    print("\n[2/7] Analyzing sequence lengths...")
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
    print("\n[3/7] Encoding sequences...")
    X_train_encoded = encode_sequences(X_train, max_length)
    X_val_encoded = encode_sequences(X_val, max_length)
    X_test_encoded = encode_sequences(X_test, max_length)
    
    # Extract features
    print("\n[4/7] Extracting engineered features...")
    X_train_features = extract_all_features(X_train)
    X_val_features = extract_all_features(X_val)
    X_test_features = extract_all_features(X_test)
    
    print(f"Extracted {X_train_features.shape[1]} features per sequence")
    
    # Normalize features
    print("\n[5/7] Normalizing features...")
    X_train_features_norm, feature_scaler = normalize_features(X_train_features)
    X_val_features_norm, _ = normalize_features(X_val_features, feature_scaler)
    X_test_features_norm, _ = normalize_features(X_test_features, feature_scaler)
    
    # Build model
    print("\n[6/7] Building model...")
    feature_dim = X_train_features_norm.shape[1]
    model = build_model(max_length, VOCAB_SIZE, feature_dim)
    print(model.summary())
    
    # Train model
    print("\n[7/7] Training model...")
    history = train_model(
        model, X_train_encoded, X_train_features_norm, y_train,
        X_val_encoded, X_val_features_norm, y_val,
        batch_size=32,
        epochs=50,
        model_save_path='amp_model.keras'
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, X_test_encoded, X_test_features_norm, y_test)
    
    print("\n" + "=" * 60)
    print("Test Set Performance Metrics")
    print("=" * 60)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print(f"MCC:         {metrics['mcc']:.4f}")
    print(f"AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Non-AMP   AMP")
    print(f"True Non-AMP    {metrics['confusion_matrix'][0,0]:4d}   {metrics['confusion_matrix'][0,1]:4d}")
    print(f"True AMP        {metrics['confusion_matrix'][1,0]:4d}   {metrics['confusion_matrix'][1,1]:4d}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_training_history(history, 'training_history.png')
    plot_confusion_matrix(metrics['confusion_matrix'], 'confusion_matrix.png')
    
    # Save preprocessing parameters and scaler
    preprocess_params = {
        'max_length': max_length,
        'vocab_size': VOCAB_SIZE,
        'amino_acids': AMINO_ACIDS,
        'feature_dim': feature_dim
    }
    with open('preprocess_params.json', 'w') as f:
        json.dump(preprocess_params, f, indent=2)
    print("Preprocessing parameters saved to preprocess_params.json")
    
    # Save feature scaler
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
    print("Feature scaler saved to feature_scaler.pkl")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Model saved to: amp_model.keras")
    print(f"Feature scaler saved to: feature_scaler.pkl")
    print(f"Use predict_sequence() function for inference on new sequences.")
    print(f"Note: Load feature_scaler.pkl for feature normalization during inference.")


if __name__ == "__main__":
    main()
