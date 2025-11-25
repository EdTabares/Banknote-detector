"""
===============================================================================
SISTEMA DE PREDICCIÓN DE BILLETES FALSOS
Usa los modelos entrenados para hacer predicciones en tiempo real
===============================================================================
"""

import numpy as np
import pickle
import json
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

class BanknotePredictor:
    """
    Sistema completo de predicción para autenticación de billetes
    """
    
    def __init__(self):
        """Inicializa el sistema de predicción"""
        self.scaler = None
        self.lr_model = None
        self.nn_model = None
        self.feature_names = ['variance', 'skewness', 'curtosis', 'entropy']
        
    def load_models(self):
        """Carga los modelos entrenados"""
        try:
            # Cargar scaler
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Scaler cargado")
            
            # Cargar Regresión Logística
            with open('logistic_regression_model.pkl', 'rb') as f:
                self.lr_model = pickle.load(f)
            print("✅ Modelo de Regresión Logística cargado")
            
            # Cargar Red Neuronal
            self.nn_model = keras.models.load_model('best_nn_model.h5')
            print("✅ Modelo de Red Neuronal cargado")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Error: No se encontró el archivo {e.filename}")
            print("   Por favor, ejecuta primero el script de entrenamiento")
            return False
        except Exception as e:
            print(f"❌ Error al cargar modelos: {e}")
            return False
    
    def predict(self, variance, skewness, curtosis, entropy, verbose=True):
        """
        Realiza predicción con ambos modelos
        
        Args:
            variance (float): Varianza de la imagen
            skewness (float): Asimetría
            curtosis (float): Curtosis
            entropy (float): Entropía
            verbose (bool): Mostrar resultados detallados
            
        Returns:
            dict: Resultados de ambos modelos
        """
        # Preparar datos
        features = np.array([[variance, skewness, curtosis, entropy]])
        features_scaled = self.scaler.transform(features)
        
        # Predicción con Regresión Logística
        lr_pred = self.lr_model.predict(features_scaled)[0]
        lr_proba = self.lr_model.predict_proba(features_scaled)[0]
        
        # Predicción con Red Neuronal
        nn_proba = self.nn_model.predict(features_scaled, verbose=0)[0, 0]
        nn_pred = 1 if nn_proba >= 0.5 else 0
        
        results = {
            'features': {
                'variance': variance,
                'skewness': skewness,
                'curtosis': curtosis,
                'entropy': entropy
            },
            'logistic_regression': {
                'prediction': int(lr_pred),
                'class': 'Falso' if lr_pred == 1 else 'Auténtico',
                'probability_fake': float(lr_proba[1]),
                'probability_authentic': float(lr_proba[0]),
                'confidence': float(max(lr_proba))
            },
            'neural_network': {
                'prediction': int(nn_pred),
                'class': 'Falso' if nn_pred == 1 else 'Auténtico',
                'probability_fake': float(nn_proba),
                'probability_authentic': float(1 - nn_proba),
                'confidence': float(max(nn_proba, 1 - nn_proba))
            }
        }
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        """Imprime los resultados de forma legible"""
        print("\n" + "="*70)
        print("RESULTADOS DE LA PREDICCIÓN")
        print("="*70)
        
        print("\n📊 Características del Billete:")
        for feature, value in results['features'].items():
            print(f"   • {feature.capitalize():12s}: {value:8.4f}")
        
        print("\n🔵 REGRESIÓN LOGÍSTICA:")
        lr = results['logistic_regression']
        icon = "✅" if lr['prediction'] == 0 else "❌"
        print(f"   {icon} Predicción: {lr['class']} (clase {lr['prediction']})")
        print(f"   • Probabilidad de ser auténtico: {lr['probability_authentic']*100:6.2f}%")
        print(f"   • Probabilidad de ser falso:     {lr['probability_fake']*100:6.2f}%")
        print(f"   • Confianza de la predicción:    {lr['confidence']*100:6.2f}%")
        
        print("\n🔴 RED NEURONAL:")
        nn = results['neural_network']
        icon = "✅" if nn['prediction'] == 0 else "❌"
        print(f"   {icon} Predicción: {nn['class']} (clase {nn['prediction']})")
        print(f"   • Probabilidad de ser auténtico: {nn['probability_authentic']*100:6.2f}%")
        print(f"   • Probabilidad de ser falso:     {nn['probability_fake']*100:6.2f}%")
        print(f"   • Confianza de la predicción:    {nn['confidence']*100:6.2f}%")
        
        # Consenso
        if lr['prediction'] == nn['prediction']:
            print(f"\n🎯 CONSENSO: Ambos modelos coinciden → Billete {lr['class'].upper()}")
        else:
            print(f"\n⚠️  DESACUERDO: Los modelos no coinciden")
            print(f"   • RL predice: {lr['class']}")
            print(f"   • RN predice: {nn['class']}")
        
        print("\n" + "="*70)
    
    def batch_predict(self, data):
        """
        Predice múltiples billetes a la vez
        
        Args:
            data (list): Lista de listas con [variance, skewness, curtosis, entropy]
            
        Returns:
            list: Lista de resultados
        """
        results = []
        for i, features in enumerate(data):
            print(f"\n📄 Analizando billete #{i+1}...")
            result = self.predict(*features, verbose=False)
            results.append(result)
            
            # Resumen corto
            lr_class = result['logistic_regression']['class']
            nn_class = result['neural_network']['class']
            match = "✅" if lr_class == nn_class else "⚠️"
            print(f"   {match} RL: {lr_class} | RN: {nn_class}")
        
        return results


def main():
    """Función principal - ejemplos de uso"""
    
    print("="*70)
    print("SISTEMA DE PREDICCIÓN DE BILLETES FALSOS")
    print("="*70)
    
    # Crear predictor
    predictor = BanknotePredictor()
    
    # Cargar modelos
    if not predictor.load_models():
        return
    
    print("\n" + "="*70)
    print("EJEMPLOS DE PREDICCIÓN")
    print("="*70)
    
    # Ejemplo 1: Billete auténtico típico
    print("\n" + "▶"*35)
    print("EJEMPLO 1: BILLETE AUTÉNTICO TÍPICO")
    print("▶"*35)
    predictor.predict(
        variance=3.6216,
        skewness=8.6661,
        curtosis=-2.8073,
        entropy=-0.44699
    )
    
    # Ejemplo 2: Billete falso típico
    print("\n" + "▶"*35)
    print("EJEMPLO 2: BILLETE FALSO TÍPICO")
    print("▶"*35)
    predictor.predict(
        variance=-1.3971,
        skewness=-4.0789,
        curtosis=5.1048,
        entropy=1.6872
    )
    
    # Ejemplo 3: Caso limítrofe
    print("\n" + "▶"*35)
    print("EJEMPLO 3: CASO LIMÍTROFE (DUDOSO)")
    print("▶"*35)
    predictor.predict(
        variance=0.5,
        skewness=1.2,
        curtosis=0.8,
        entropy=-0.1
    )
    
    # Ejemplo 4: Batch prediction
    print("\n" + "▶"*35)
    print("EJEMPLO 4: PREDICCIÓN EN LOTE (BATCH)")
    print("▶"*35)
    
    batch_data = [
        [3.6216, 8.6661, -2.8073, -0.44699],  # Auténtico
        [-1.3971, -4.0789, 5.1048, 1.6872],   # Falso
        [2.5831, 6.9155, -1.9433, -0.89617],  # Auténtico
        [-2.3434, -6.4835, 8.3451, 2.1567],   # Falso
    ]
    
    results = predictor.batch_predict(batch_data)
    
    # Estadísticas del batch
    print("\n" + "="*70)
    print("ESTADÍSTICAS DEL LOTE")
    print("="*70)
    
    lr_authentic = sum(1 for r in results if r['logistic_regression']['prediction'] == 0)
    nn_authentic = sum(1 for r in results if r['neural_network']['prediction'] == 0)
    consensus = sum(1 for r in results if r['logistic_regression']['prediction'] == r['neural_network']['prediction'])
    
    print(f"\n   Total de billetes analizados: {len(results)}")
    print(f"\n   🔵 Regresión Logística:")
    print(f"      • Auténticos: {lr_authentic}")
    print(f"      • Falsos:     {len(results) - lr_authentic}")
    print(f"\n   🔴 Red Neuronal:")
    print(f"      • Auténticos: {nn_authentic}")
    print(f"      • Falsos:     {len(results) - nn_authentic}")
    print(f"\n   🎯 Consenso: {consensus}/{len(results)} ({consensus/len(results)*100:.1f}%)")
    
    # Modo interactivo
    print("\n" + "="*70)
    print("MODO INTERACTIVO")
    print("="*70)
    print("\n¿Deseas analizar tus propios billetes? (s/n): ", end="")
    
    try:
        response = input().lower()
        
        while response == 's':
            print("\n" + "-"*70)
            print("Ingresa las características del billete:")
            
            try:
                variance = float(input("  Varianza:  "))
                skewness = float(input("  Asimetría: "))
                curtosis = float(input("  Curtosis:  "))
                entropy = float(input("  Entropía:  "))
                
                predictor.predict(variance, skewness, curtosis, entropy)
                
            except ValueError:
                print("\n❌ Error: Ingresa valores numéricos válidos")
            
            print("\n¿Analizar otro billete? (s/n): ", end="")
            response = input().lower()
            
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido por el usuario")
    
    print("\n" + "="*70)
    print("✅ PROGRAMA FINALIZADO")
    print("="*70)


if __name__ == "__main__":
    main()