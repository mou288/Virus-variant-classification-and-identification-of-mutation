import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os




print("TensorFlow version:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PARENT_DIR, "data", "processed_3way_split")

RESULT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading preprocessed data...")


X_train = np.load(f"{DATA_DIR}/train_embeddings.npy", mmap_mode="r")
X_val   = np.load(f"{DATA_DIR}/val_embeddings.npy", mmap_mode="r")
X_test  = np.load(f"{DATA_DIR}/test_embeddings.npy", mmap_mode="r")

y_train_labels = pd.read_csv(f"{DATA_DIR}/train_labels.csv")["label"]
y_val_labels   = pd.read_csv(f"{DATA_DIR}/val_labels.csv")["label"]
y_test_labels  = pd.read_csv(f"{DATA_DIR}/test_labels.csv")["label"]


encoder = LabelEncoder()
y_train_int = encoder.fit_transform(y_train_labels)
y_val_int   = encoder.transform(y_val_labels)
y_test_int  = encoder.transform(y_test_labels)

joblib.dump(encoder, os.path.join(RESULT_DIR, "label_encoder.joblib"))

NUM_CLASSES = len(encoder.classes_)

y_train_cat = to_categorical(y_train_int, NUM_CLASSES)
y_val_cat   = to_categorical(y_val_int, NUM_CLASSES)
y_test_cat  = to_categorical(y_test_int, NUM_CLASSES)

# class weights
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
class_weights_dict = dict(zip(np.unique(y_train_int), weights))


INPUT_TIMESTEPS = X_train.shape[1]
INPUT_FEATURES  = X_train.shape[2]

model = Sequential([
    # CNN BLOCK 1
    Conv1D(128, 7, padding='same', activation='relu',
           input_shape=(INPUT_TIMESTEPS, INPUT_FEATURES)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.25),

    # CNN BLOCK 2
    Conv1D(256, 5, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.25),

    # BiLSTM LAYER 
    Bidirectional(LSTM(128, return_sequences=False, dropout=0.3)),
    Dropout(0.3),

    # Dense classifier
    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(NUM_CLASSES, activation='softmax')
])


optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4,
    clipnorm=1.0
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

model.summary()


callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint(os.path.join(RESULT_DIR, "best_model.keras"), save_best_only=True, monitor='val_loss'),
    CSVLogger(os.path.join(RESULT_DIR, "training_log.csv"))
]


batch_size = 64

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=15,
    batch_size=batch_size,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    shuffle=True
)


plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend(); plt.title("Accuracy")
plt.savefig(os.path.join(RESULT_DIR, "accuracy_curve.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend(); plt.title("Loss")
plt.savefig(os.path.join(RESULT_DIR, "loss_curve.png"))
plt.close()

print("\nSaved training graphs!")


# TEST EVALUATION

test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print("\nTest Accuracy:", test_acc)

y_pred = np.argmax(model.predict(X_test), axis=1)

# Classification report
report = classification_report(y_test_int, y_pred,
                               target_names=encoder.classes_,
                               zero_division=0)

with open(os.path.join(RESULT_DIR, "test_report.txt"), "w") as f:
    f.write(report)

print("\nClassification Report:\n", report)


# CONFUSION MATRIX

cm = confusion_matrix(y_test_int, y_pred)
cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)

plt.figure(figsize=(12, 9))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()


# PER-CLASS ACCURACY

per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
per_class_df = pd.DataFrame({
    "Class": encoder.classes_,
    "Accuracy": per_class_accuracy
})
per_class_df.to_csv(os.path.join(RESULT_DIR, "per_class_accuracy.csv"), index=False)

plt.figure(figsize=(14, 7))
sns.barplot(x=encoder.classes_, y=per_class_accuracy)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.title("Per-Class Accuracy")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig(os.path.join(RESULT_DIR, "per_class_accuracy.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\n🎉 All results generated in /results/")