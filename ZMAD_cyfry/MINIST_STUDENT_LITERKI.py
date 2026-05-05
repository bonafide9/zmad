#Damian Białek

import warnings
warnings.filterwarnings("ignore")
import os
import gzip
import glob
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



FILES = {
    "train_images": (
        "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte.gz",
    ),
    "train_labels": (
        "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ),
    "test_images": (
        "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ),
    "test_labels": (
        "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ),
}

def download(url: str, filename: str) -> None:
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Plik już istnieje: {filename} (rozmiar: {os.path.getsize(filename)} B)")
        return

    print(f"Pobieram {url} -> {filename}")
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        print(f"Błąd podczas pobierania {url}: {e}")
        raise

def load_mnist(data_dir: str = "mnist_data"):
    os.makedirs(data_dir, exist_ok=True)

    for key, (url, fname) in FILES.items():
        path = os.path.join(data_dir, fname)
        download(url, path)

    def read_images(path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2051:
                raise ValueError(f"Zły magic number w {path}: {magic}")
            n_images = int.from_bytes(f.read(4), "big")
            n_rows = int.from_bytes(f.read(4), "big")
            n_cols = int.from_bytes(f.read(4), "big")
            data = f.read(n_images * n_rows * n_cols)
            images = np.frombuffer(data, dtype=np.uint8)
            images = images.reshape(n_images, n_rows * n_cols)
            return images

    def read_labels(path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2049:
                raise ValueError(f"Zły magic number w {path}: {magic}")
            n_labels = int.from_bytes(f.read(4), "big")
            data = f.read(n_labels)
            labels = np.frombuffer(data, dtype=np.uint8)
            return labels

    train_images_path = os.path.join(data_dir, FILES["train_images"][1])
    train_labels_path = os.path.join(data_dir, FILES["train_labels"][1])
    test_images_path = os.path.join(data_dir, FILES["test_images"][1])
    test_labels_path = os.path.join(data_dir, FILES["test_labels"][1])

    X_train = read_images(train_images_path)
    y_train = read_labels(train_labels_path)
    X_test = read_images(test_images_path)
    y_test = read_labels(test_labels_path)

    return (X_train, y_train), (X_test, y_test)


def preprocess_image_to_canvas28(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    arr = 255 - arr
    thresh = 80
    bin_arr = (arr > thresh).astype(np.uint8)

    coords = np.argwhere(bin_arr == 1)
    if coords.size == 0:
        raise ValueError(f"Nie znalazłem żadnych jasnych pikseli w {image_path}")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    digit = arr[y0:y1, x0:x1]

    digit_img = Image.fromarray(digit.astype(np.uint8))
    digit_img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    digit_resized = np.array(digit_img)

    h, w = digit_resized.shape
    canvas = np.zeros((28, 28), dtype=np.float32)
    y_off = (28 - h) // 2
    x_off = (28 - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = digit_resized.astype(np.float32)

    canvas /= 255.0
    return canvas


POLISH_LABEL_MAP = {
    "zero": 0, "jeden": 1, "dwa": 2, "trzy": 3, "cztery": 4,
    "piec": 5, "szesc": 6, "siedem": 7, "osiem": 8, "dziewiec": 9,
}

LETTER_LABEL_MAP = {
    "a": "A", "b": "B", "c": "C", "d": "D", "e": "E", "f": "F",
    "g": "G", "h": "H", "i": "I", "j": "J", "k": "K", "l": "L",
    "m": "M", "n": "N", "o": "O", "p": "P", "q": "Q", "r": "R",
    "s": "S", "t": "T", "u": "U", "v": "V", "w": "W", "x": "X",
    "y": "Y", "z": "Z"
}

def load_custom_data(base_dir: str, label_map: dict):
    X_list = []
    y_list = []

    if not os.path.isdir(base_dir):
        print(f"(Informacja) Brak katalogu {base_dir} – pomijam wczytywanie.")
        return np.empty((0, 784), dtype=np.float32), np.empty((0,), dtype=object)

    for dirname, label in label_map.items():
        folder = os.path.join(base_dir, dirname)
        if not os.path.isdir(folder):
            continue

        pattern = os.path.join(folder, "*.*")
        for path in glob.glob(pattern):
            if not path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            try:
                canvas = preprocess_image_to_canvas28(path)
                X_list.append(canvas.reshape(-1))
                y_list.append(label)
            except Exception as e:
                print(f"Pomijam {path}: {e}")

    if not X_list:
        print(f"(Informacja) Nie znaleziono żadnych obrazów w {base_dir}")
        return np.empty((0, 784), dtype=np.float32), np.empty((0,), dtype=object)

    X_custom = np.stack(X_list).astype(np.float32)
    y_custom = np.array(y_list) # Automatycznie dostosuje typ do int lub string
    print(f"Wczytano {len(X_custom)} przykładów z {base_dir}")
    return X_custom, y_custom


def show_examples_from_test(X: np.ndarray, y_true: np.ndarray, model: MLPClassifier, n_samples: int = 9):
    if X.size == 0:
        print("Brak danych testowych w tym trybie.")
        return

    n_samples = min(n_samples, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)

    plt.figure(figsize=(8, 8))
    for i, idx in enumerate(indices, start=1):
        img = X[idx].reshape(28, 28)
        true_label = y_true[idx]

        x_single = X[idx].reshape(1, -1)
        pred_label = model.predict(x_single)[0]

        plt.subplot(3, 3, i)
        plt.imshow(img, cmap="gray")
        plt.title(f"Prawda: {true_label}\nPred: {pred_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def predict_custom_image(model: MLPClassifier, image_path: str):
    canvas = preprocess_image_to_canvas28(image_path)
    x_flat = canvas.reshape(1, -1)

    pred_label = model.predict(x_flat)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_flat)[0]
    else:
        proba = None

    return pred_label, proba, canvas

def main():
    print("Wybierz tryb treningu sieci:")
    print("[1] MNIST + moje cyfry (katalog 'moje_cyfry/')")
    print("[2] Tylko moje cyfry (bez MNIST)")
    print("[3] Tylko moje litery (katalog 'moje_litery/')")
    mode = input("Tryb [1/2/3, domyślnie 1]: ").strip()
    if mode not in {"1", "2", "3"}:
        mode = "1"

    has_mnist_test = False

    if mode == "1":
        print("\n== TRYB 1: Trening na MNIST + własne cyfry ==")
        (X_train, y_train), (X_test, y_test) = load_mnist(data_dir="mnist_data")
        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0
        has_mnist_test = True

        X_custom, y_custom = load_custom_data("moje_cyfry", POLISH_LABEL_MAP)
        if X_custom.shape[0] > 0:
            X_train = np.vstack([X_train, X_custom])
            y_train = np.hstack([y_train, y_custom])

    elif mode == "2":
        print("\n== TRYB 2: Trening tylko na własnych cyfrach ==")
        X_train, y_train = load_custom_data("moje_cyfry", POLISH_LABEL_MAP)
        if X_train.shape[0] == 0:
            return
        X_test = np.empty((0, 784), dtype=np.float32)
        y_test = np.empty((0,), dtype=object)

    elif mode == "3":
        print("\n== TRYB 3: Trening tylko na własnych literach ==")
        X_train, y_train = load_custom_data("moje_litery", LETTER_LABEL_MAP)
        if X_train.shape[0] == 0:
            print("Brak danych do treningu. Upewnij się, że katalog 'moje_litery/' istnieje i ma odpowiednie podfoldery.")
            return
        X_test = np.empty((0, 784), dtype=np.float32)
        y_test = np.empty((0,), dtype=object)

    # Definicja sieci
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=20,
        random_state=42,
        verbose=True,
    )

    print("\n== Trening MLP na wybranym zbiorze treningowym ==")
    mlp.fit(X_train, y_train)
    print("KLASY MODELU:", mlp.classes_)

    if has_mnist_test:
        print("\n== Ewaluacja na zbiorze testowym MNIST ==")
        y_pred = mlp.predict(X_test)
        print(f"Accuracy (dokładność) na zbiorze testowym: {accuracy_score(y_test, y_pred):.4f}\n")

    while True:
        print("\n=======================================")
        print("MENU:")
        if has_mnist_test:
            print("[1] Pokaż przykładowe dane testowe MNIST z predykcjami")
        else:
            print("[1] (Niedostępne w tym trybie)")
        print("[2] Rozpoznaj symbol z własnego pliku graficznego")
        print("[3] Zakończ program")
        print("=======================================")

        choice = input("Wybierz opcję (1/2/3): ").strip()

        if choice == "1":
            if not has_mnist_test: continue
            show_examples_from_test(X_test, y_test, mlp)

        elif choice == "2":
            image_path = input("Podaj pełną ścieżkę do pliku z obrazem:\n> ").strip()
            if not os.path.isfile(image_path):
                print("Plik nie istnieje.")
                continue

            try:
                pred_label, proba, arr = predict_custom_image(mlp, image_path)
                print(f"\nSieć przewiduje znak: {pred_label}")
                if proba is not None:
                    print("Prawdopodobieństwa klas:")
                    for cls, p in zip(mlp.classes_, proba):
                        print(f"  {cls}: {p:.3f}")

                plt.figure()
                plt.imshow(arr, cmap="gray")
                plt.title(f"Przewidywany znak: {pred_label}")
                plt.axis("off")
                plt.show()

            except Exception as e:
                print(f"Wystąpił błąd: {e}")

        elif choice == "3":
            print("Zamykanie programu.")
            break

if __name__ == "__main__":
    main()