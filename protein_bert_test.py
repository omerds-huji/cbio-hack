import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from proteinbert import (
    OutputType,
    OutputSpec,
    FinetuningModelGenerator,
    load_pretrained_model,
)
from proteinbert.conv_and_global_attention_model import (
    get_model_with_hidden_layers_as_outputs,
)

# ==========================================
# 1. Data Loading & Preparation
# ==========================================


def parse_fasta_file(file_path: str):
    """
    Parse a FASTA file into a list of sequences
    """
    sequences = []
    with open(file_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences.append(str(record.seq))
    return sequences


print("Loading data...")

amp_train_seqs = parse_fasta_file("dataset/AMP.tr.fa")
decoy_train_seqs = parse_fasta_file("dataset/DECOY.tr.fa")
train_seqs = amp_train_seqs + decoy_train_seqs
train_labels = np.concatenate(
    [np.ones(len(amp_train_seqs)), np.zeros(len(decoy_train_seqs))]
)

# --- Load Test ---
amp_test_seqs = parse_fasta_file("dataset/AMP.te.fa")
decoy_test_seqs = parse_fasta_file("dataset/DECOY.te.fa")
test_seqs = amp_test_seqs + decoy_test_seqs
test_labels = np.concatenate(
    [np.ones(len(amp_test_seqs)), np.zeros(len(decoy_test_seqs))]
)

# --- Load Eval (Validation) ---
amp_eval_seqs = parse_fasta_file("dataset/AMP.eval.fa")
decoy_eval_seqs = parse_fasta_file("dataset/DECOY.eval.fa")
eval_seqs = amp_eval_seqs + decoy_eval_seqs
eval_labels = np.concatenate(
    [np.ones(len(amp_eval_seqs)), np.zeros(len(decoy_eval_seqs))]
)

print(
    f"Data loaded: {len(train_seqs)} Train, {len(eval_seqs)} Validation, {len(test_seqs)} Test."
)

# ==========================================
# 2. Prepare ProteinBERT Model
# ==========================================

all_sequences = train_seqs + eval_seqs + test_seqs
max_seq_len = max([len(s) for s in all_sequences]) + 2

# Load the input encoder
pretrained_model_generator, input_encoder = load_pretrained_model(".")

# Encode Data
# Note: encode_X handles the list of strings automatically
x_train = input_encoder.encode_X(train_seqs, max_seq_len)
y_train = np.array(train_labels)

x_eval = input_encoder.encode_X(eval_seqs, max_seq_len)
y_eval = np.array(eval_labels)

x_test = input_encoder.encode_X(test_seqs, max_seq_len)
y_test = np.array(test_labels)

# Define Output Spec (Binary Classification)
output_type = OutputType(False, "binary")
output_spec = OutputSpec(output_type)

# Create the Fine-tuning Model
model_generator = FinetuningModelGenerator(
    pretraining_model_generator=pretrained_model_generator,
    output_spec=output_spec,
    pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
    dropout_rate=0.5,
)

training_model = model_generator.create_model(max_seq_len)

# ==========================================
# 3. Train (Fine-tune)
# ==========================================
print("Starting training...")

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

training_model.fit(
    x_train,
    y_train,
    validation_data=(
        x_eval,
        y_eval,
    ),
    epochs=10,  # Increased epochs, early stopping will handle it
    batch_size=32,
    callbacks=[early_stopping],
)

# ==========================================
# 4. Evaluation & Metrics Calculation
# ==========================================


def calculate_metrics(model, x_input, y_true, method_name="ProteinBERT"):
    """
    Calculates SENS, SPEC, ACC, MCC, and auROC.
    """
    print(f"Calculating metrics for {method_name}...")

    # Predict probabilities
    y_pred_probs = model.predict(x_input)

    y_prob_positive = y_pred_probs.ravel()  # Flatten to 1D array
    y_pred = (y_prob_positive > 0.5).astype(int)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrics
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        auroc = roc_auc_score(y_true, y_prob_positive)
    except ValueError:
        auroc = 0.0

    results = {
        "Method": method_name,
        "SENS(%)": round(sens * 100, 2),
        "SPEC(%)": round(spec * 100, 2),
        "ACC(%)": round(acc * 100, 2),
        "MCC": round(mcc, 4),
        "auROC(%)": round(auroc * 100, 2),
    }

    return results, y_pred


# Generate Results on the held-out TEST set
results, y_pred = calculate_metrics(
    training_model, x_test, y_test, method_name="ProteinBERT"
)

# ==========================================
# 5. Display Output
# ==========================================
df_results = pd.DataFrame([results])

print("\n--- Final Evaluation on Test Set ---")
print(df_results.to_string(index=False))

# df_results.to_csv("proteinbert_results.csv", index=False)


def get_aa_composition(sequences):
    """
    Calculates the frequency of each amino acid in a list of sequences.
    """
    total_counts = Counter()
    total_len = 0
    for seq in sequences:
        total_counts.update(seq)
        total_len += len(seq)

    # Avoid division by zero
    if total_len == 0:
        return {}

    # Normalize to frequency
    composition = {k: v / total_len for k, v in total_counts.items()}
    return composition


print("\n--- Generating Amino Acid Composition Analysis ---")

groups = {"TP": [], "TN": [], "FP": [], "FN": []}

for seq, true_l, pred_l in zip(test_seqs, y_test, y_pred):
    if true_l == 1 and pred_l == 1:
        groups["TP"].append(seq)
    elif true_l == 0 and pred_l == 0:
        groups["TN"].append(seq)
    elif true_l == 0 and pred_l == 1:
        groups["FP"].append(seq)
    elif true_l == 1 and pred_l == 0:
        groups["FN"].append(seq)

print(
    f"Counts: TP={len(groups['TP'])}, TN={len(groups['TN'])}, FP={len(groups['FP'])}, FN={len(groups['FN'])}"
)

# 3. Calculate Compositions
plot_data = []
# Define the amino acids of interest (or all of them)
target_aas = sorted(list(set(''.join(test_seqs))))

for group_name, seqs in groups.items():
    comp = get_aa_composition(seqs)
    for aa in target_aas:
        plot_data.append(
            {"Amino Acid": aa, "Frequency": comp.get(aa, 0), "Group": group_name}
        )

df_plot = pd.DataFrame(plot_data)

# 4. Plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_plot,
    x="Amino Acid",
    y="Frequency",
    hue="Group",
    palette={"TP": "green", "TN": "gray", "FP": "orange", "FN": "red"},
)
plt.title("Amino Acid Composition by Classification Outcome")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("error_analysis_composition.png")
print("Plot saved to 'error_analysis_composition.png'")
