import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import scipy.io
import os

# --- 1. CONFIGURAZIONE ---
FILE_PATH = os.path.join('dataset', 'emnist-letters.mat')

# Le classi che ci interessano: B(2), G(7), T(20), Y(25)
TARGET_INDICES = [2, 7, 20, 25]
CLASS_NAMES = ['Blue', 'Green', 'Testina', 'Yellow']

# --- 2. CARICAMENTO DATI ---
def load_and_prep_data(path):
    print(f"[INFO] Caricamento dataset da: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File non trovato! Controlla che esista: {path}")

    mat = scipy.io.loadmat(path)
    dataset = mat['dataset']

    X_train = dataset['train'][0,0]['images'][0,0]
    y_train = dataset['train'][0,0]['labels'][0,0]
    X_test  = dataset['test'][0,0]['images'][0,0]
    y_test  = dataset['test'][0,0]['labels'][0,0]

    # Reshape + transpose (EMNIST è ruotato/specchiato nel .mat)
    X_train = X_train.reshape((-1, 28, 28)).transpose(0, 2, 1)
    X_test  = X_test.reshape((-1, 28, 28)).transpose(0, 2, 1)

    # Filtraggio: teniamo solo B, G, T, Y
    print("[INFO] Filtraggio classi (B, G, T, Y)...")
    train_mask = np.isin(y_train, TARGET_INDICES).flatten()
    test_mask  = np.isin(y_test,  TARGET_INDICES).flatten()

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test,  y_test  = X_test[test_mask],   y_test[test_mask]

    # Rimappatura etichette: (2,7,20,25) -> (0,1,2,3)
    mapping = {old: new for new, old in enumerate(TARGET_INDICES)}
    y_train = np.array([mapping[val[0]] for val in y_train])
    y_test  = np.array([mapping[val[0]] for val in y_test])

    # Normalizzazione [0,255] -> [0,1]
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0

    # Aggiunta canale (scala di grigi): shape -> (N, 28, 28, 1)
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    # One-Hot Encoding
    y_train_cat = utils.to_categorical(y_train, 4)
    y_test_cat  = utils.to_categorical(y_test,  4)

    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[INFO] Distribuzione classi train: { {CLASS_NAMES[i]: int(np.sum(y_train==i)) for i in range(4)} }")

    return X_train, y_train_cat, X_test, y_test_cat


# --- 3. COSTRUZIONE DELLA CNN ---
def build_cnn():
    model = models.Sequential(name="CNN_coloring_project")

    # Input
    model.add(layers.Input(shape=(28, 28, 1)))

    # -------------------------------------------------------
    # DATA AUGMENTATION
    # Attiva SOLO durante il training, ignorata in predict/evaluate.
    #
    # Parametri scelti per lettere scritte a mano (28x28 px):
    #   - RandomRotation(0.06): ±~22° — simula inclinazione naturale
    #     della scrittura.
    #   - RandomZoom((-0.15, 0.05)): zoom out max 15%, zoom in max 5%.
    #     Asimmetrico: è più comune che una lettera sia piccola/lontana
    #     che ingrandita oltre i bordi della cella.
    #   - RandomTranslation(0.08, 0.08): shift ±8% = ~±2px.
    #     Simula centratura imperfetta senza rischiare di tagliare il tratto.
    # -------------------------------------------------------
    model.add(layers.RandomRotation(0.06))
    model.add(layers.RandomZoom((-0.15, 0.05)))
    model.add(layers.RandomTranslation(0.08, 0.08))

    # -------------------------------------------------------
    # ARCHITETTURA CONVOLUZIONALE
    # Blocco 1: 32 filtri — cattura tratti semplici (bordi, curve)
    # -------------------------------------------------------
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))       # 28x28 -> 14x14

    # Blocco 2: 64 filtri — combina i tratti in strutture più complesse
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))       # 14x14 -> 7x7

    # Blocco 3: 128 filtri — rappresentazioni ad alto livello
    # MODIFICA: aumentato da 64 a 128 filtri per avere più capacità
    # espressiva nell'ultimo blocco conv (prima del classificatore).
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    # Nessun MaxPooling qui: 7x7 è già piccolo, un ulteriore pooling
    # ridurrebbe troppo la feature map (3x3) con rischio di perdita info.

    # -------------------------------------------------------
    # CLASSIFICATORE FINALE
    # -------------------------------------------------------
    model.add(layers.Flatten())                  # 7x7x128 = 6272 features
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))               # Riduce overfitting
    model.add(layers.Dense(4, activation='softmax'))

    # Optimizer: Adam con learning rate leggermente abbassato
    # (default 1e-3 può essere aggressivo con augmentation)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# --- 4. MAIN ---
try:
    # A. Caricamento
    X_train, y_train, X_test, y_test = load_and_prep_data(FILE_PATH)

    # B. Modello
    model = build_cnn()
    model.summary()


    # C. Training
    print("[INFO] Inizio training...")
    history = model.fit(
        X_train, y_train,
        epochs=8,
        batch_size=64,
        validation_data=(X_test, y_test),
    )

    # E. Grafici Accuracy e Loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],     label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()

    # F. Valutazione + Matrice di Confusione
    print("\n[INFO] Valutazione finale sul test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[INFO] Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    y_pred         = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true         = np.argmax(y_test, axis=1)

    print("\n" + classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.title('Matrice di Confusione')
    plt.tight_layout()
    plt.show()

    # G. Salvataggio
    model.save('cnn_coloring_project.keras')
    print("\n[SUCCESSO] Modello salvato come 'cnn_coloring_project.keras'")

except Exception as e:
    import traceback
    print(f"\n[ERRORE] {e}")
    traceback.print_exc()