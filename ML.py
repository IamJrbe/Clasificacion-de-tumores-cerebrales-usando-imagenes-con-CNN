import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import ImageTk, Image
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Parámetros principales para el procesamiento y entrenamiento
IMG_SIZE = 128  # Tamaño al que se redimensionan las imágenes (128x128)
DATASET_PATH = r'C:\Users\Jr\Desktop\Ciencia\data'  # Ruta del dataset de imágenes
BATCH_SIZE = 32  # Tamaño del batch para entrenamiento
EPOCHS = 20  # Número de épocas para entrenar el modelo
CLASSES = ['glioma', 'meningioma', 'pituitary', 'no']  # Clases de tumores y 'no' tumor

# === FUNCIÓN PARA CARGAR IMÁGENES Y SUS ETIQUETAS ===
def load_images_multiclass():
    X = []  # Lista para almacenar imágenes
    y = []  # Lista para almacenar etiquetas (índices de clase)
    for idx, clase in enumerate(CLASSES):
        carpeta_clase = os.path.join(DATASET_PATH, clase)  # Carpeta para cada clase
        for archivo in os.listdir(carpeta_clase):
            ruta = os.path.join(carpeta_clase, archivo)
            img = cv2.imread(ruta)  # Leer imagen con OpenCV
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Redimensionar imagen
                X.append(img)
                y.append(idx)  # Guardar índice de la clase correspondiente
    X = np.array(X)  # Convertir a arreglo numpy para facilitar el procesamiento
    y = to_categorical(y, num_classes=len(CLASSES))  # One-hot encoding para las etiquetas
    return X, y

# === FUNCIÓN PARA CLASIFICAR UNA IMAGEN DE PACIENTE Y GENERAR REPORTE EN PDF ===
def clasificar_paciente_con_datos(ruta_imagen):
    try:
        # Solicitar datos del paciente
        nombre = input("Nombre del paciente: ")
        fecha_nacimiento = input("Fecha de nacimiento (YYYY-MM-DD): ")
        motivo = input("Motivo de consulta: ")
        genero = input("Género (M/F): ")
        fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Fecha y hora actual

        # Carpeta donde se guardarán los reportes PDF
        carpeta_salida = r"C:\Users\Jr\Desktop\Ciencia\data\pacientes"
        os.makedirs(carpeta_salida, exist_ok=True)  # Crear carpeta si no existe

        # Obtener siguiente número para nombrar archivos (ejemplo: paciente 1, paciente 2, ...)
        archivos = os.listdir(carpeta_salida)
        numeros = [int(re.search(r"paciente (\d+)", f).group(1))
                   for f in archivos if re.search(r"paciente (\d+)", f)]
        siguiente_num = max(numeros, default=0) + 1
        nombre_archivo = f"paciente {siguiente_num}"

        # Leer y preparar la imagen para clasificación
        img = cv2.imread(ruta_imagen)
        if img is None:
            print(f"No se pudo leer la imagen: {ruta_imagen}")
            return
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_norm = img_resized / 255.0  # Normalizar píxeles entre 0 y 1
        img_input = np.expand_dims(img_norm, axis=0)  # Añadir dimensión batch

        # Predecir clase y obtener la probabilidad
        pred = model.predict(img_input)[0]
        clase_idx = np.argmax(pred)  # Clase con mayor probabilidad
        prob = pred[clase_idx]

        # Crear título según resultado de la predicción
        titulo = "No se detectó tumor" if CLASSES[clase_idx] == "no" else f"Tumor detectado: {CLASSES[clase_idx]} ({prob:.2f})"

        # Guardar imágenes y gráficas para incluir en el PDF
        img_path = os.path.join(carpeta_salida, f"{nombre_archivo}_imagen.png")
        grafico_path = os.path.join(carpeta_salida, f"{nombre_archivo}_grafico.png")

        # Convertir imagen a RGB para mostrar con matplotlib
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(titulo)
        plt.axis('off')
        plt.savefig(img_path)
        plt.close()

        # Gráfica de barras con la confianza por clase
        plt.figure()
        sns.barplot(x=CLASSES, y=pred)
        plt.ylim([0, 1])
        plt.title('Confianza por clase')
        plt.savefig(grafico_path)
        plt.close()

        # Crear PDF con los datos del paciente y resultados
        pdf_path = os.path.join(carpeta_salida, f"{nombre_archivo}.pdf")
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 750, "Reporte de Clasificación de Tumor Cerebral")

        # Escribir datos del paciente y resultado
        c.setFont("Helvetica", 12)
        c.drawString(50, 720, f"Nombre: {nombre}")
        c.drawString(50, 700, f"Fecha de nacimiento: {fecha_nacimiento}")
        c.drawString(50, 680, f"Género: {genero}")
        c.drawString(50, 660, f"Motivo de consulta: {motivo}")
        c.drawString(50, 640, f"Fecha de análisis: {fecha_actual}")
        c.drawString(50, 620, f"Resultado: {titulo}")

        # Insertar imagen y gráfico en el PDF
        c.drawImage(ImageReader(img_path), 50, 380, width=200, height=200)
        c.drawImage(ImageReader(grafico_path), 300, 380, width=250, height=200)

        # Mostrar confianza por clase en texto
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 340, "Confianza por clase:")
        c.setFont("Helvetica", 12)
        y_line = 320
        for i, clase in enumerate(CLASSES):
            c.drawString(70, y_line, f"{clase}: {pred[i]:.2f}")
            y_line -= 20

        c.save()  # Guardar PDF

        # Eliminar imágenes temporales
        os.remove(img_path)
        os.remove(grafico_path)

        print(f"\nPDF generado: {pdf_path}")

    except Exception as e:
        print(f"Error: {e}")

# Cargar todas las imágenes y etiquetas
X, y = load_images_multiclass()
X = X / 255.0  # Normalizar imágenes a rango [0, 1]

# Dividir datos en entrenamiento (64%), validación (16%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data augmentation para aumentar la variedad de datos de entrenamiento
datagen = ImageDataGenerator(
    rotation_range=20,       # Rotación aleatoria hasta 20 grados
    zoom_range=0.15,         # Zoom aleatorio hasta 15%
    width_shift_range=0.15,  # Desplazamiento horizontal hasta 15%
    height_shift_range=0.15, # Desplazamiento vertical hasta 15%
    horizontal_flip=True,    # Voltear horizontalmente
    fill_mode='nearest'      # Rellenar píxeles vacíos con el valor más cercano
)
datagen.fit(X_train_final)

# Generador para usar data augmentation durante el entrenamiento
train_generator = datagen.flow(X_train_final, y_train_final, batch_size=BATCH_SIZE)

# Generador para validación sin aumentos (normalización solo)
val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# Modelo de red neuronal convolucional para clasificación de imágenes
model = Sequential([
    # Primera capa convolucional: 32 filtros 3x3, activación ReLU, padding 'same' para mantener tamaño
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    # Normalización por lotes para acelerar el entrenamiento y estabilizar
    BatchNormalization(),
    # Capa de max pooling para reducir la resolución a la mitad (2x2)
    MaxPooling2D(2, 2),

    # Segunda capa convolucional: 64 filtros 3x3, activación ReLU, padding 'same'
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Tercera capa convolucional: 128 filtros 3x3, activación ReLU, padding 'same'
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Cuarta capa convolucional: 256 filtros 3x3, activación ReLU, padding 'same'
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Aplanar las características extraídas para conectarlas con la capa densa
    Flatten(),
    # Capa densa con 512 neuronas y activación ReLU para aprender representaciones complejas
    Dense(512, activation='relu'),
    # Dropout para evitar sobreajuste, con tasa de 0.5 (50% de neuronas se apagan al azar)
    Dropout(0.5),
    # Capa de salida con número de neuronas igual a la cantidad de clases, activación softmax para clasificación múltiple
    Dense(len(CLASSES), activation='softmax')
])
# Compilar con tasa de aprendizaje levemente menor
model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
checkpoint = ModelCheckpoint('mejor_modelo_multiclase.h5', monitor='val_loss', save_best_only=True)

# Entrenamiento
history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator,
                    callbacks=[early_stopping, checkpoint], verbose=2)
# Evaluación
model = load_model("mejor_modelo_multiclase.h5")
loss, acc = model.evaluate(X_test, y_test)
print(f"\nPrecisión en test: {acc*100:.2f}%")

# Reporte
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred, target_names=CLASSES))
print("Matriz de confusión:\n", confusion_matrix(y_true, y_pred))

def entrenar_modelo(progress_callback=None):
    global model, X, y, BATCH_SIZE

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train_final)
    train_generator = datagen.flow(X_train_final, y_train_final, batch_size=BATCH_SIZE)

    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

    epochs = 10

    # Entrenamiento con callback para actualizar progreso en GUI
    for epoch in range(epochs):
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=1,
            verbose=0
        )
        if progress_callback:
            progress_callback(epoch + 1, epochs)

def clasificar_con_tkinter():
    ctk.set_appearance_mode("dark")  # Modos: "dark", "light", "system"
    ctk.set_default_color_theme("blue")  # Temas: "blue", "dark-blue", "green"

    def seleccionar_imagen():
        archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.jpeg")])
        if archivo:
            ruta_imagen.set(archivo)

    def borrar_datos():
        entrada_nombre.delete(0, ctk.END)
        entrada_fecha.delete(0, ctk.END)
        entrada_motivo.delete(0, ctk.END)
        entrada_operador.delete(0, ctk.END)
        combo_genero.set("")
        ruta_imagen.set("")

    def generar_pdf():
        try:
            nombre = entrada_nombre.get()
            fecha_nac = entrada_fecha.get()
            motivo = entrada_motivo.get()
            genero = combo_genero.get()
            operador = entrada_operador.get()
            imagen = ruta_imagen.get()

            if not all([nombre, fecha_nac, motivo, genero, operador, imagen]):
                messagebox.showerror("Error", "Todos los campos son obligatorios.")
                return

            fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            carpeta_salida = r"C:\Users\Jr\Desktop\Ciencia\data\pacientes"
            os.makedirs(carpeta_salida, exist_ok=True)

            archivos = os.listdir(carpeta_salida)
            numeros = [int(re.search(r"paciente (\d+)", f).group(1))
                       for f in archivos if re.search(r"paciente (\d+)", f)]
            siguiente_num = max(numeros, default=0) + 1
            nombre_archivo = f"paciente {siguiente_num}"

            img = cv2.imread(imagen)
            if img is None:
                messagebox.showerror("Error", f"No se pudo leer la imagen: {imagen}")
                return
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_norm = img_resized / 255.0
            img_input = np.expand_dims(img_norm, axis=0)

            pred = model.predict(img_input)[0]
            clase_idx = np.argmax(pred)
            prob = pred[clase_idx]
            titulo = "No se detectó tumor" if CLASSES[clase_idx] == "no" else f"Tumor detectado: {CLASSES[clase_idx]} ({prob:.2f})"

            img_path = os.path.join(carpeta_salida, f"{nombre_archivo}_imagen.png")
            grafico_path = os.path.join(carpeta_salida, f"{nombre_archivo}_grafico.png")

            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(titulo)
            plt.axis('off')
            plt.savefig(img_path)
            plt.close()

            plt.figure(figsize=(6,4))
            sns.barplot(x=CLASSES, y=pred)
            plt.ylim([0, 1])
            plt.title('Confianza por clase')
            plt.savefig(grafico_path)
            plt.close()

            pdf_path = os.path.join(carpeta_salida, f"{nombre_archivo}.pdf")
            c = canvas.Canvas(pdf_path, pagesize=letter)
            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, 750, "Reporte de Clasificación de Tumor Cerebral")

            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, 720, "Datos del Paciente:")
            c.setFont("Helvetica", 12)
            c.drawString(70, 700, f"Nombre: {nombre}")
            c.drawString(70, 680, f"Fecha de nacimiento: {fecha_nac}")
            c.drawString(70, 660, f"Género: {genero}")
            c.drawString(70, 640, f"Motivo de consulta: {motivo}")

            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, 610, "Datos del Operador:")
            c.setFont("Helvetica", 12)
            c.drawString(70, 590, f"Operador que toma el estudio: {operador}")

            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, 560, f"Fecha de análisis: {fecha_actual}")
            c.drawString(50, 540, f"Resultado: {titulo}")

            c.drawImage(ImageReader(img_path), 50, 300, width=200, height=200)
            c.drawImage(ImageReader(grafico_path), 300, 300, width=250, height=200)

            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, 270, "Confianza por clase:")
            c.setFont("Helvetica", 12)
            y_line = 250
            for i, clase in enumerate(CLASSES):
                c.drawString(70, y_line, f"{clase}: {pred[i]:.2f}")
                y_line -= 20

            c.save()

            os.remove(img_path)
            os.remove(grafico_path)

            messagebox.showinfo("Éxito", f"PDF generado: {pdf_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    ventana = ctk.CTk()
    ventana.title("Clasificación de Tumor Cerebral")
    ventana.geometry("650x450")
    ventana.minsize(600, 400)

    # Etiquetas y entradas
    ctk.CTkLabel(ventana, text="Nombre del paciente:").grid(row=0, column=0, padx=15, pady=10, sticky="e")
    entrada_nombre = ctk.CTkEntry(ventana, width=300)
    entrada_nombre.grid(row=0, column=1, padx=15, pady=10)

    ctk.CTkLabel(ventana, text="Fecha de nacimiento (YYYY-MM-DD):").grid(row=1, column=0, padx=15, pady=10, sticky="e")
    entrada_fecha = ctk.CTkEntry(ventana, width=300)
    entrada_fecha.grid(row=1, column=1, padx=15, pady=10)

    ctk.CTkLabel(ventana, text="Motivo de consulta:").grid(row=2, column=0, padx=15, pady=10, sticky="e")
    entrada_motivo = ctk.CTkEntry(ventana, width=300)
    entrada_motivo.grid(row=2, column=1, padx=15, pady=10)

    ctk.CTkLabel(ventana, text="Género:").grid(row=3, column=0, padx=15, pady=10, sticky="e")
    combo_genero = ctk.CTkComboBox(ventana, values=["M", "F"], width=300)
    combo_genero.grid(row=3, column=1, padx=15, pady=10)

    ctk.CTkLabel(ventana, text="Nombre del operador:").grid(row=4, column=0, padx=15, pady=10, sticky="e")
    entrada_operador = ctk.CTkEntry(ventana, width=300)
    entrada_operador.grid(row=4, column=1, padx=15, pady=10)

    ctk.CTkLabel(ventana, text="Imagen:").grid(row=5, column=0, padx=15, pady=10, sticky="e")
    ruta_imagen = ctk.StringVar()
    entrada_imagen = ctk.CTkEntry(ventana, textvariable=ruta_imagen, width=300)
    entrada_imagen.grid(row=5, column=1, padx=15, pady=10)

    boton_seleccionar = ctk.CTkButton(ventana, text="Seleccionar imagen", command=seleccionar_imagen)
    boton_seleccionar.grid(row=5, column=2, padx=10, pady=10)

    boton_clasificar = ctk.CTkButton(ventana, text="Clasificar y generar PDF", command=generar_pdf)
    boton_clasificar.grid(row=6, column=0, columnspan=3, pady=15, sticky="ew", padx=20)

    boton_borrar = ctk.CTkButton(ventana, text="Borrar datos", command=borrar_datos)
    boton_borrar.grid(row=7, column=0, columnspan=3, pady=5, sticky="ew", padx=20)

    ventana.grid_columnconfigure(1, weight=1)


    ventana.mainloop()

if __name__ == "__main__":
    clasificar_con_tkinter()

def clasificar_paciente_por_consola():
    while True:
        print("\n--- Clasificación por consola ---")
        ruta = input("Ruta de la imagen (o 'salir' para terminar): ")
        if ruta.lower() == "salir":
            break
        clasificar_paciente_con_datos(ruta)


def clasificar_paciente_con_datos(ruta_imagen):
    try:
        nombre = input("Nombre del paciente: ")
        fecha_nacimiento = input("Fecha de nacimiento (YYYY-MM-DD): ")
        motivo = input("Motivo de consulta: ")
        genero = input("Género (M/F): ")
        fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        carpeta_salida = r"C:\Users\Jr\Desktop\Ciencia\data\pacientes"
        os.makedirs(carpeta_salida, exist_ok=True)

        archivos = os.listdir(carpeta_salida)
        numeros = [int(re.search(r"paciente (\d+)", f).group(1))
                   for f in archivos if re.search(r"paciente (\d+)", f)]
        siguiente_num = max(numeros, default=0) + 1
        nombre_archivo = f"paciente {siguiente_num}"

        img = cv2.imread(ruta_imagen)
        if img is None:
            print(f"No se pudo leer la imagen: {ruta_imagen}")
            return
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_norm = img_resized / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        pred = model.predict(img_input)[0]
        clase_idx = np.argmax(pred)
        prob = pred[clase_idx]

        titulo = "No se detectó tumor" if CLASSES[clase_idx] == "no" else f"Tumor detectado: {CLASSES[clase_idx]} ({prob:.2f})"

        img_path = os.path.join(carpeta_salida, f"{nombre_archivo}_imagen.png")
        grafico_path = os.path.join(carpeta_salida, f"{nombre_archivo}_grafico.png")

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(titulo)
        plt.axis('off')
        plt.savefig(img_path)
        plt.close()

        plt.figure()
        sns.barplot(x=CLASSES, y=pred)
        plt.ylim([0, 1])
        plt.title('Confianza por clase')
        plt.savefig(grafico_path)
        plt.close()

        pdf_path = os.path.join(carpeta_salida, f"{nombre_archivo}.pdf")
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 750, "Reporte de Clasificación de Tumor Cerebral")

        c.setFont("Helvetica", 12)
        c.drawString(50, 720, f"Nombre: {nombre}")
        c.drawString(50, 700, f"Fecha de nacimiento: {fecha_nacimiento}")
        c.drawString(50, 680, f"Género: {genero}")
        c.drawString(50, 660, f"Motivo de consulta: {motivo}")
        c.drawString(50, 640, f"Fecha de análisis: {fecha_actual}")
        c.drawString(50, 620, f"Resultado: {titulo}")

        c.drawImage(ImageReader(img_path), 50, 380, width=200, height=200)
        c.drawImage(ImageReader(grafico_path), 300, 380, width=250, height=200)

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 340, "Confianza por clase:")
        c.setFont("Helvetica", 12)
        y_line = 320
        for i, clase in enumerate(CLASSES):
            c.drawString(70, y_line, f"{clase}: {pred[i]:.2f}")
            y_line -= 20

        c.save()

        os.remove(img_path)
        os.remove(grafico_path)

        print(f"\nPDF generado: {pdf_path}")

    except Exception as e:
        print(f"Error: {e}")

# === INICIO ===
clasificar_con_tkinter()
img_height = 64
img_width = 64

