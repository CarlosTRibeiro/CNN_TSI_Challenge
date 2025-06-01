# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score

# Definir e compilar o modelo CNN
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Carregar os CSVs
df_rgb_8x8 = pd.read_csv('rgb_8x8.csv')
df_gray_8x8 = pd.read_csv('gray_8x8.csv')
df_rgb_28x28 = pd.read_csv('rgb_28x28.csv')
df_gray_28x28 = pd.read_csv('gray_28x28.csv')
df_meta = pd.read_csv('meta.csv')

# Verificar se o número de amostras é consistente
assert df_rgb_8x8.shape[0] == df_meta.shape[0]
assert df_gray_8x8.shape[0] == df_meta.shape[0]
assert df_rgb_28x28.shape[0] == df_meta.shape[0]
assert df_gray_28x28.shape[0] == df_meta.shape[0]

# Preparar os rótulos
le = LabelEncoder()
y_encoded = le.fit_transform(df_meta['malignancy'])
y_categorical = to_categorical(y_encoded)

# Normalização dos dados de imagem
X_rgb_8x8 = df_rgb_8x8.values.reshape(-1, 8, 8, 3) / 255.0
X_gray_8x8 = df_gray_8x8.values.reshape(-1, 8, 8, 1) / 255.0
X_rgb_28x28 = df_rgb_28x28.values.reshape(-1, 28, 28, 3) / 255.0
X_gray_28x28 = df_gray_28x28.values.reshape(-1, 28, 28, 1) / 255.0

# Lista com os datasets para iterar
datasets = {
    'RGB 8x8': (X_rgb_8x8, (8, 8, 3)),
    'Gray 8x8': (X_gray_8x8, (8, 8, 1)),
    'RGB 28x28': (X_rgb_28x28, (28, 28, 3)),
    'Gray 28x28': (X_gray_28x28, (28, 28, 1))
}

# Função para plotar precisão e perda
def plot_accuracy_loss(history, title):
    plt.figure(figsize=(12, 5))

    # Precisão
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisão - Treino')
    plt.plot(history.history['val_accuracy'], label='Precisão - Validação')
    plt.title(f'Precisão ao Longo das Épocas ({title})')
    plt.xlabel('Épocas')
    plt.ylabel('Precisão')
    plt.legend()

    # Perda
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda - Treino')
    plt.plot(history.history['val_loss'], label='Perda - Validação')
    plt.title(f'Perda ao Longo das Épocas ({title})')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Função para plotar a Curva ROC
def plot_roc_curve(y_true, y_pred_probs, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - {title}')
    plt.legend(loc="lower right")
    plt.show()

# Função para plotar a Matriz de Confusão
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Previsões')
    plt.ylabel('Valores Reais')
    plt.title(f'Matriz de Confusão - {title}')
    plt.show()

# Função para plotar Sensibilidade e Especificidade
def plot_sensitivity_specificity(y_true, y_pred, title):
    sensitivity = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Sensibilidade', 'Especificidade'], y=[sensitivity, specificity], palette='viridis')
    plt.ylim(0, 1)
    plt.title(f'Sensibilidade vs Especificidade ({title})')
    plt.ylabel('Score')
    plt.show()

# Função para plotar a distribuição das probabilidades de malignidade
def plot_prediction_probabilities(y_pred_probs, title):
    plt.figure(figsize=(8, 6))
    sns.histplot(y_pred_probs[:, 1], bins=20, kde=True, color='purple')
    plt.title(f'Distribuição das Probabilidades de Malignidade ({title})')
    plt.xlabel('Probabilidade de Maligno')
    plt.ylabel('Frequência')
    plt.show()

# Função para visualizar algumas imagens dos datasets
def plot_images(X, title, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        if X.shape[-1] == 1:
            plt.imshow(X[i].reshape(X.shape[1], X.shape[2]), cmap='gray')
        else:
            plt.imshow(X[i])
        plt.axis('off')
    plt.suptitle(f'Exemplo de Imagens - {title}', fontsize=16)
    plt.show()

# Treinamento dos modelos e geração de gráficos para cada dataset
histories = {}

for name, (X_data, input_shape) in datasets.items():
    print(f"\nTreinando modelo para: {name}")

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_categorical, test_size=0.2, random_state=42)
    
    # Criar e compilar o modelo
    model = create_cnn_model(input_shape)
    
    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)
    histories[name] = history
    
    # Avaliação no conjunto de teste
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Resultado no conjunto de teste para {name}:")
    print(f"Precisão: {test_accuracy:.4f}, Perda: {test_loss:.4f}")
    
    # Gerar Previsões
    y_pred_probs = model.predict(X_test)
    y_pred_classes = y_pred_probs.argmax(axis=1)
    y_true_classes = y_test.argmax(axis=1)

    # Gerar gráficos
    plot_accuracy_loss(history, name)
    plot_roc_curve(y_true_classes, y_pred_probs, name)
    plot_confusion_matrix(y_true_classes, y_pred_classes, classes=['Benigno', 'Maligno'], title=name)
    plot_sensitivity_specificity(y_true_classes, y_pred_classes, title=name)
    plot_prediction_probabilities(y_pred_probs, title=name)
    plot_images(X_test, f'Imagens do Conjunto de Teste - {name}')

print("\nTreinamento e geração de gráficos concluídos para todos os datasets!")
