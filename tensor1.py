import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Definir o caminho para a pasta de imagens
dataset_dir = 'img'  # Substitua pelo caminho para sua pasta de imagens

# Definir parâmetros
img_height, img_width = 224, 224  # Definindo um novo tamanho para as imagens
batch_size = 32

# Carregar o conjunto de dados a partir do diretório
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'  # Converter para escala de cinza
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'  # Converter para escala de cinza
)

# Normalizar os valores dos pixels para o intervalo [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Definir o modelo da CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),  # Alterar para 1 canal
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(train_dataset.class_names), activation='softmax')  # Número de classes baseado nas pastas
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(normalized_train_ds, epochs=10, 
                    validation_data=normalized_val_ds)

# Avaliar a precisão do modelo com os dados de validação
test_loss, test_acc = model.evaluate(normalized_val_ds)
print("Validation accuracy:", test_acc)

# Plotar a precisão de treinamento e validação ao longo das épocas
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
