
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def cargar_dataset(base_dir, emociones_validas, image_size=(48, 48), grayscale=True):
    X, y = [], []
    for emocion in emociones_validas:
        carpeta = os.path.join(base_dir, emocion)
        if not os.path.exists(carpeta):
            print(f"Advertencia: no se encontró la carpeta {carpeta}")
            continue
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(('.jpg', '.png', '.jpeg')):
                path_img = os.path.join(carpeta, archivo)
                img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                X.append(img)
                y.append(emocion)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    emociones_finales = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    print("Cargando dataset1...")
    X1, y1 = cargar_dataset('dataset1', emociones_finales)

    print("Cargando dataset2...")
    X2, y2 = cargar_dataset('dataset2', emociones_finales)

    print("Fusionando datasets...")
    X_total = np.concatenate([X1, X2], axis=0)
    y_total = np.concatenate([y1, y2], axis=0)

    print("Preprocesando datos...")
    X_total = X_total / 255.0
    X_total = X_total.reshape(-1, 48, 48, 1)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_total)
    y_cat = to_categorical(y_encoded)

    print("Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_total, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Guardando datos procesados...")
    np.savez_compressed("datos_emociones.npz",
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test,
                        clases=le.classes_)

    print("Proceso completado. Dataset fusionado guardado en 'datos_emociones_fusionados.npz'")
