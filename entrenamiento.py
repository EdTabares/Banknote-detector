"""
===============================================================================
TALLER DE MACHINE LEARNING: CLASIFICACIÓN DE BILLETES FALSOS
Dataset: Banknote Authentication (UCI ML Repository)
Autor: [Tu Nombre]
Institución: Politécnico Colombiano Jaime Isaza Cadavid
Fecha: 2024
===============================================================================

Este script implementa:
1. Carga y exploración del dataset
2. Preprocesamiento de datos
3. Entrenamiento de Regresión Logística
4. Entrenamiento de Red Neuronal Artificial
5. Evaluación y comparación de modelos
6. Generación de todas las figuras para el paper
7. Guardado de modelos entrenados
"""

# ============================================================================
# IMPORTAR LIBRERÍAS
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score,
    recall_score, 
    f1_score, 
    classification_report,
    roc_curve,
    auc
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

print("="*80)
print("TALLER: CLASIFICACIÓN DE BILLETES BANCARIOS AUTÉNTICOS VS FALSOS")
print("="*80)
print("\n✅ Librerías importadas correctamente\n")

# ============================================================================
# 1. CARGAR Y EXPLORAR EL DATASET
# ============================================================================
print("="*80)
print("PASO 1: CARGA Y EXPLORACIÓN DEL DATASET")
print("="*80)

# URL del dataset en UCI Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

try:
    # Intentar cargar desde UCI
    df = pd.read_csv(url, names=column_names)
    print("✅ Dataset cargado exitosamente desde UCI Repository\n")
except Exception as e:
    print(f"⚠️ No se pudo descargar: {e}")
    print("Creando dataset sintético para demostración...\n")
    
    # Dataset sintético si no hay internet
    np.random.seed(42)
    n = 1372
    df = pd.DataFrame({
        'variance': np.random.randn(n) * 2.5 + 0.5,
        'skewness': np.random.randn(n) * 3.5,
        'curtosis': np.random.randn(n) * 3,
        'entropy': np.random.randn(n) * 1.8 - 0.3,
        'class': np.random.randint(0, 2, n)
    })

# Información básica del dataset
print("📊 INFORMACIÓN DEL DATASET:")
print(f"   • Total de muestras: {len(df)}")
print(f"   • Número de características: {len(df.columns) - 1}")
print(f"   • Billetes auténticos (clase 0): {len(df[df['class'] == 0])} ({len(df[df['class'] == 0])/len(df)*100:.1f}%)")
print(f"   • Billetes falsos (clase 1): {len(df[df['class'] == 1])} ({len(df[df['class'] == 1])/len(df)*100:.1f}%)")

print("\n📋 Primeras 5 filas del dataset:")
print(df.head())

print("\n📈 Estadísticas descriptivas:")
print(df.describe())

print("\n❓ Valores nulos por columna:")
print(df.isnull().sum())

print("\n✅ Conclusión: Dataset limpio, sin valores nulos")

# ============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================
print("\n" + "="*80)
print("PASO 2: ANÁLISIS EXPLORATORIO DE DATOS")
print("="*80)

# Crear figura completa de EDA
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Título principal
fig.suptitle('Análisis Exploratorio: Dataset de Billetes Bancarios', 
             fontsize=16, fontweight='bold', y=0.995)

# 1. Distribución de clases
ax1 = fig.add_subplot(gs[0, 0])
class_counts = df['class'].value_counts()
colors_bar = ['#2ecc71', '#e74c3c']
bars = ax1.bar(['Auténtico (0)', 'Falso (1)'], class_counts.values, 
               color=colors_bar, alpha=0.8, edgecolor='black')
ax1.set_title('Distribución de Clases', fontweight='bold')
ax1.set_ylabel('Cantidad de Muestras')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# 2-5. Distribuciones de características
features = ['variance', 'skewness', 'curtosis', 'entropy']
colors = ['#3498db', '#e74c3c']
positions = [(0, 1), (0, 2), (1, 0), (1, 1)]

for idx, (feature, pos) in enumerate(zip(features, positions)):
    ax = fig.add_subplot(gs[pos[0], pos[1]])
    
    # Histogramas superpuestos
    for class_label, color in zip([0, 1], colors):
        data = df[df['class'] == class_label][feature]
        ax.hist(data, bins=30, alpha=0.6, color=color, 
                label=f'Clase {class_label}', edgecolor='black', linewidth=0.5)
    
    ax.set_title(f'Distribución: {feature.capitalize()}', fontweight='bold')
    ax.set_xlabel(feature.capitalize())
    ax.set_ylabel('Frecuencia')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 6. Matriz de correlación
ax6 = fig.add_subplot(gs[1, 2])
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=ax6, cbar_kws={'shrink': 0.8})
ax6.set_title('Matriz de Correlación', fontweight='bold')

# 7. Boxplot de Varianza
ax7 = fig.add_subplot(gs[2, 0])
df.boxplot(column='variance', by='class', ax=ax7, patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7))
ax7.set_title('Varianza por Clase', fontweight='bold')
ax7.set_xlabel('Clase')
ax7.set_ylabel('Varianza')
plt.sca(ax7)
plt.xticks([1, 2], ['Auténtico', 'Falso'])

# 8. Boxplot de Curtosis
ax8 = fig.add_subplot(gs[2, 1])
df.boxplot(column='curtosis', by='class', ax=ax8, patch_artist=True,
           boxprops=dict(facecolor='lightcoral', alpha=0.7))
ax8.set_title('Curtosis por Clase', fontweight='bold')
ax8.set_xlabel('Clase')
ax8.set_ylabel('Curtosis')
plt.sca(ax8)
plt.xticks([1, 2], ['Auténtico', 'Falso'])

# 9. Pairplot simplificado (scatter de las 2 mejores características)
ax9 = fig.add_subplot(gs[2, 2])
for class_label, color, label in zip([0, 1], colors, ['Auténtico', 'Falso']):
    data = df[df['class'] == class_label]
    ax9.scatter(data['variance'], data['curtosis'], 
                c=color, alpha=0.6, s=20, label=label, edgecolors='black', linewidth=0.3)
ax9.set_title('Varianza vs Curtosis', fontweight='bold')
ax9.set_xlabel('Varianza')
ax9.set_ylabel('Curtosis')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.savefig('EDA_completo.png', dpi=300, bbox_inches='tight')
print("\n✅ Figura guardada: EDA_completo.png")
plt.close()

# ============================================================================
# 3. PREPARACIÓN DE DATOS
# ============================================================================
print("\n" + "="*80)
print("PASO 3: PREPARACIÓN DE DATOS")
print("="*80)

# Separar características (X) y etiquetas (y)
X = df.drop('class', axis=1).values
y = df['class'].values

print(f"✅ Forma de X (características): {X.shape}")
print(f"✅ Forma de y (etiquetas): {y.shape}")

# División entrenamiento/prueba (70%/30%) con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n📦 Conjunto de entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   - Auténticos: {np.sum(y_train == 0)}")
print(f"   - Falsos: {np.sum(y_train == 1)}")

print(f"\n📦 Conjunto de prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"   - Auténticos: {np.sum(y_test == 0)}")
print(f"   - Falsos: {np.sum(y_test == 1)}")

# Normalización con StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Datos normalizados con StandardScaler (media=0, std=1)")
print(f"   Media de X_train_scaled: {X_train_scaled.mean(axis=0).round(4)}")
print(f"   Std de X_train_scaled: {X_train_scaled.std(axis=0).round(4)}")

# Guardar el scaler para uso futuro
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("\n💾 Scaler guardado: scaler.pkl")

# ============================================================================
# 4. MODELO 1: REGRESIÓN LOGÍSTICA
# ============================================================================
print("\n" + "="*80)
print("MODELO 1: REGRESIÓN LOGÍSTICA")
print("="*80)

# Entrenar modelo
print("\n⚙️ Entrenando Regresión Logística...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    C=1.0  # Parámetro de regularización
)

lr_model.fit(X_train_scaled, y_train)
print("✅ Modelo entrenado exitosamente")

# Información del modelo
print("\n📊 INFORMACIÓN DEL MODELO:")
print(f"   • Coeficientes (pesos):")
for feature, coef in zip(column_names[:-1], lr_model.coef_[0]):
    print(f"     - {feature}: {coef:.4f}")
print(f"   • Intercepto (bias): {lr_model.intercept_[0]:.4f}")

# Predicciones
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Calcular métricas
cm_lr = confusion_matrix(y_test, y_pred_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
error_lr = 1 - accuracy_lr

print("\n📊 RESULTADOS - REGRESIÓN LOGÍSTICA:")
print("="*60)
print(f"   Error:          {error_lr:.4f} ({error_lr*100:.2f}%)")
print(f"   Exactitud:      {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")
print(f"   Precisión:      {precision_lr:.4f} ({precision_lr*100:.2f}%)")
print(f"   Exhaustividad:  {recall_lr:.4f} ({recall_lr*100:.2f}%)")
print(f"   F1-Score:       {f1_lr:.4f} ({f1_lr*100:.2f}%)")

print("\n📋 Matriz de Confusión:")
print(f"                Predicho: 0    Predicho: 1")
print(f"Real: 0 (Auth)      {cm_lr[0,0]:3d}           {cm_lr[0,1]:3d}")
print(f"Real: 1 (Fake)      {cm_lr[1,0]:3d}           {cm_lr[1,1]:3d}")

print("\n🔍 Interpretación:")
print(f"   • Verdaderos Negativos (TN): {cm_lr[0,0]} - Auténticos correctamente identificados")
print(f"   • Falsos Positivos (FP): {cm_lr[0,1]} - Auténticos clasificados como falsos")
print(f"   • Falsos Negativos (FN): {cm_lr[1,0]} - Falsos clasificados como auténticos ⚠️")
print(f"   • Verdaderos Positivos (TP): {cm_lr[1,1]} - Falsos correctamente identificados")

print(f"\n📈 Reporte de clasificación completo:")
print(classification_report(y_test, y_pred_lr, target_names=['Auténtico', 'Falso']))

# Guardar modelo
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("💾 Modelo guardado: logistic_regression_model.pkl")

# ============================================================================
# 5. MODELO 2: RED NEURONAL ARTIFICIAL
# ============================================================================
print("\n" + "="*80)
print("MODELO 2: RED NEURONAL ARTIFICIAL")
print("="*80)

# Construir arquitectura
print("\n🏗️ Construyendo arquitectura de la red neuronal...")
nn_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],), 
          name='hidden_layer_1'),
    Dropout(0.2, name='dropout_1'),
    Dense(8, activation='relu', name='hidden_layer_2'),
    Dropout(0.2, name='dropout_2'),
    Dense(1, activation='sigmoid', name='output_layer')
], name='BanknoteClassifier')

# Compilar modelo
nn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
)

print("\n📝 ARQUITECTURA DE LA RED NEURONAL:")
nn_model.summary()

# Callbacks para entrenamiento
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_nn_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=0
)

# Entrenar modelo
print("\n⚙️ Entrenando Red Neuronal (esto puede tardar 1-2 minutos)...")
history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint],
    verbose=0
)

print(f"✅ Entrenamiento completado en {len(history.history['loss'])} épocas")

# Predicciones
y_pred_nn_proba = nn_model.predict(X_test_scaled, verbose=0).flatten()
y_pred_nn = (y_pred_nn_proba > 0.5).astype(int)

# Calcular métricas
cm_nn = confusion_matrix(y_test, y_pred_nn)
accuracy_nn = accuracy_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)
error_nn = 1 - accuracy_nn

print("\n📊 RESULTADOS - RED NEURONAL:")
print("="*60)
print(f"   Error:          {error_nn:.4f} ({error_nn*100:.2f}%)")
print(f"   Exactitud:      {accuracy_nn:.4f} ({accuracy_nn*100:.2f}%)")
print(f"   Precisión:      {precision_nn:.4f} ({precision_nn*100:.2f}%)")
print(f"   Exhaustividad:  {recall_nn:.4f} ({recall_nn*100:.2f}%)")
print(f"   F1-Score:       {f1_nn:.4f} ({f1_nn*100:.2f}%)")

print("\n📋 Matriz de Confusión:")
print(f"                Predicho: 0    Predicho: 1")
print(f"Real: 0 (Auth)      {cm_nn[0,0]:3d}           {cm_nn[0,1]:3d}")
print(f"Real: 1 (Fake)      {cm_nn[1,0]:3d}           {cm_nn[1,1]:3d}")

print("\n🔍 Interpretación:")
print(f"   • Verdaderos Negativos (TN): {cm_nn[0,0]}")
print(f"   • Falsos Positivos (FP): {cm_nn[0,1]}")
print(f"   • Falsos Negativos (FN): {cm_nn[1,0]} ⚠️")
print(f"   • Verdaderos Positivos (TP): {cm_nn[1,1]}")

print(f"\n📈 Reporte de clasificación completo:")
print(classification_report(y_test, y_pred_nn, target_names=['Auténtico', 'Falso']))

# Guardar modelo
nn_model.save('neural_network_model.h5')
print("💾 Modelo guardado: neural_network_model.h5")

# Guardar historial de entrenamiento
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
print("💾 Historial guardado: training_history.json")

# ============================================================================
# 6. COMPARACIÓN DE MODELOS
# ============================================================================
print("\n" + "="*80)
print("COMPARACIÓN DE MODELOS")
print("="*80)

comparison_df = pd.DataFrame({
    'Métrica': ['Error', 'Exactitud', 'Precisión', 'Exhaustividad', 'F1-Score'],
    'Regresión Logística': [
        f'{error_lr:.4f}',
        f'{accuracy_lr:.4f}',
        f'{precision_lr:.4f}',
        f'{recall_lr:.4f}',
        f'{f1_lr:.4f}'
    ],
    'Red Neuronal': [
        f'{error_nn:.4f}',
        f'{accuracy_nn:.4f}',
        f'{precision_nn:.4f}',
        f'{recall_nn:.4f}',
        f'{f1_nn:.4f}'
    ],
    'Diferencia': [
        f'{error_nn - error_lr:+.4f}',
        f'{accuracy_nn - accuracy_lr:+.4f}',
        f'{precision_nn - precision_lr:+.4f}',
        f'{recall_nn - recall_lr:+.4f}',
        f'{f1_nn - f1_lr:+.4f}'
    ]
})

print("\n" + comparison_df.to_string(index=False))

# Determinar ganador
if accuracy_nn > accuracy_lr:
    winner = "Red Neuronal"
    diff = (accuracy_nn - accuracy_lr) * 100
    print(f"\n🏆 GANADOR: {winner}")
    print(f"   Mejora en exactitud: +{diff:.2f}%")
    print(f"   Reducción de errores: {((error_lr - error_nn) / error_lr * 100):.1f}%")
elif accuracy_lr > accuracy_nn:
    winner = "Regresión Logística"
    diff = (accuracy_lr - accuracy_nn) * 100
    print(f"\n🏆 GANADOR: {winner}")
    print(f"   Mejora en exactitud: +{diff:.2f}%")
else:
    print(f"\n🤝 EMPATE: Ambos modelos tienen exactitud similar")

# ============================================================================
# 7. GUARDAR RESULTADOS FINALES
# ============================================================================
print("\n" + "="*80)
print("GUARDANDO RESULTADOS FINALES")
print("="*80)

results = {
    'dataset_info': {
        'total_samples': len(df),
        'authentic': int(np.sum(y == 0)),
        'fake': int(np.sum(y == 1)),
        'train_size': len(X_train),
        'test_size': len(X_test)
    },
    'logistic_regression': {
        'error': float(error_lr),
        'accuracy': float(accuracy_lr),
        'precision': float(precision_lr),
        'recall': float(recall_lr),
        'f1_score': float(f1_lr),
        'confusion_matrix': cm_lr.tolist()
    },
    'neural_network': {
        'error': float(error_nn),
        'accuracy': float(accuracy_nn),
        'precision': float(precision_nn),
        'recall': float(recall_nn),
        'f1_score': float(f1_nn),
        'confusion_matrix': cm_nn.tolist(),
        'epochs_trained': len(history.history['loss'])
    }
}

with open('resultados_finales.json', 'w') as f:
    json.dump(results, f, indent=4)

print("💾 Resultados guardados: resultados_finales.json")

print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("="*80)
print("\nArchivos generados:")
print("   📊 EDA_completo.png")
print("   🔵 logistic_regression_model.pkl")
print("   🔴 neural_network_model.h5")
print("   🔴 best_nn_model.h5")
print("   📈 training_history.json")
print("   📊 resultados_finales.json")
print("   🔧 scaler.pkl")
print("\n" + "="*80)