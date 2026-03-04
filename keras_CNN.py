import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import scipy.io
import os

FILE_PATH = os.path.join('dataset', 'emnist-letters.mat')
TARGET_INDICES = [2, 7, 20, 25]
CLASS_NAMES = ['Blue', 'Green', 'Testina', 'Yellow']

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

    # Rimappatura etichette
    mapping = {old: new for new, old in enumerate(TARGET_INDICES)}
    y_train = np.array([mapping[val[0]] for val in y_train])
    y_test  = np.array([mapping[val[0]] for val in y_test])

    # Normalizzazione
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0

    X_train = X_train[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    y_train_cat = utils.to_categorical(y_train, 4)
    y_test_cat  = utils.to_categorical(y_test,  4)

    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[INFO] Distribuzione classi train: { {CLASS_NAMES[i]: int(np.sum(y_train==i)) for i in range(4)} }")

    return X_train, y_train_cat, X_test, y_test_cat


def build_cnn():
    model = models.Sequential(name="CNN_coloring_project")

    model.add(layers.Input(shape=(28, 28, 1)))

    model.add(layers.RandomRotation(0.06))
    model.add(layers.RandomZoom((-0.15, 0.05)))
    model.add(layers.RandomTranslation(0.08, 0.08))

    # ARCHITETTURA CONVOLUZIONALE
    # Blocco 1: 32 filtri — cattura tratti semplici (bordi, curve)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2))) # 28x28 -> 14x14

    # Blocco 2: 64 filtri — combina i tratti in strutture più complesse
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2))) # 14x14 -> 7x7

    # Blocco 3: 128 filtri — rappresentazioni ad alto livello
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    

    model.add(layers.Flatten()) # 7x7x64 = 3136 features
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
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

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

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
        model_name = 'cnn_coloring_project.keras'
        model.save(model_name)
        print(f"\n[SUCCESSO] Modello salvato come {model_name}")

    except Exception as e:
        import traceback
        print(f"\n[ERRORE] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()