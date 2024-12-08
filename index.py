import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

class AplicacionPrediccionVentas:
    load_dotenv()
    def __init__(self):
        # Parámetros de conexión a base de datos desde variables de entorno
        self.parametros_db = {
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'port': os.getenv('DB_PORT')
        }
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def conectar_base_datos(self):
        """Establecer conexión segura a la base de datos"""
        print(self.parametros_db['user'])
        print(self.parametros_db['password'])
        print(self.parametros_db['host'])
        print(self.parametros_db['database'])
        try:
            # Crear cadena de conexión SQLAlchemy
            cadena_conexion = f"postgresql://{self.parametros_db['user']}:{self.parametros_db['password']}@{self.parametros_db['host']}/{self.parametros_db['database']}?sslmode=require"
            
            # Validar que no haya campos vacíos
            if not all([
                self.parametros_db['host'], 
                self.parametros_db['database'], 
                self.parametros_db['user'], 
                self.parametros_db['password'],
                self.parametros_db['port']
            ]):
                raise ValueError("Faltan credenciales de conexión. Verifique su archivo .env")
            
            engine = create_engine(cadena_conexion)
            self.logger.info("Conexión a base de datos establecida exitosamente")
            return engine
        except Exception as e:
            self.logger.error(f"Error de conexión a base de datos: {e}")
            st.error(f"No se pudo conectar a la base de datos: {e}")
            return None
    
    def obtener_datos_ventas(self, engine):
        """Obtener y preprocesar datos de ventas"""
        consulta = """
        SELECT 
            "createdAt"::date as fecha_venta, 
            SUM(total) as total_ventas_diarias,
            SUM("itemsInVenta") as total_items_diarios,
            COUNT(id) as cantidad_transacciones
        FROM "Venta"
        GROUP BY "createdAt"::date
        ORDER BY fecha_venta
        """
        
        try:
            # Leer datos en DataFrame
            df = pd.read_sql(consulta, engine)
            df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
            
            # Ordenar y establecer índice
            df = df.sort_values('fecha_venta')
            df.set_index('fecha_venta', inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"Error al recuperar datos de ventas: {e}")
            st.error("No se pudieron cargar los datos de ventas")
            return None
    
    def preparar_datos_series_temporales(self, df):
        """Preparar datos para predicción de series temporales"""
        try:
            # Remuestrear a datos mensuales
            df_mensual = df.resample('ME').sum()
            
            # Crear características rezagadas
            df_mensual['ventas_lag1'] = df_mensual['total_ventas_diarias'].shift(1)
            df_mensual['ventas_lag2'] = df_mensual['total_ventas_diarias'].shift(2)
            df_mensual['ventas_lag3'] = df_mensual['total_ventas_diarias'].shift(3)
            
            # Eliminar filas con valores nulos
            df_mensual.dropna(inplace=True)
            
            return df_mensual
        except Exception as e:
            self.logger.error(f"Error al preparar datos: {e}")
            st.error("Error en la preparación de datos")
            return None
    
    def entrenar_modelo_prediccion_ventas(self, df_mensual):
        """Entrenar modelo de regresión RandomForest para predicción de ventas"""
        # Preparar características y objetivo
        X = df_mensual[['ventas_lag1', 'ventas_lag2', 'ventas_lag3']]
        y = df_mensual['total_ventas_diarias']
        
        try:
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Escalar características
            escalador = StandardScaler()
            X_train_escalado = escalador.fit_transform(X_train)
            X_test_escalado = escalador.transform(X_test)
            
            # Entrenar modelo RandomForest
            modelo_rf = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                max_depth=10  # Añadido para prevenir overfitting
            )
            modelo_rf.fit(X_train_escalado, y_train)
            
            return modelo_rf, escalador
        except Exception as e:
            self.logger.error(f"Error al entrenar modelo: {e}")
            st.error("No se pudo entrenar el modelo de predicción")
            return None, None
    
    def predecir_proximos_4_meses(self, df_mensual, modelo, escalador):
        """Predecir ventas para los próximos 4 meses"""
        try:
            # Obtener últimos 3 meses como características
            ultimos_3_meses = df_mensual[['total_ventas_diarias']].tail(3)
            
            # Preparar características
            caracteristicas = ultimos_3_meses['total_ventas_diarias'].values.reshape(1, -1)
            caracteristicas_escaladas = escalador.transform(caracteristicas)
            
            # Predecir próximos 4 meses
            predicciones = []
            caracteristicas_actuales = caracteristicas_escaladas[0]
            
            for _ in range(4):
                prediccion = modelo.predict(caracteristicas_actuales.reshape(1, -1))[0]
                predicciones.append(prediccion)
                
                # Deslizar ventana
                caracteristicas_actuales = np.roll(caracteristicas_actuales, -1)
                caracteristicas_actuales[-1] = prediccion
            
            # Generar fechas futuras
            ultima_fecha = df_mensual.index[-1]
            fechas_futuras = [ultima_fecha + timedelta(days=30*(i+1)) for i in range(4)]
            
            df_predicciones = pd.DataFrame({
                'fecha': fechas_futuras,
                'ventas_predecidas': predicciones
            }).set_index('fecha')
            
            return df_predicciones
        except Exception as e:
            self.logger.error(f"Error al predecir ventas: {e}")
            st.error("No se pudieron realizar predicciones")
            return None
    
    def visualizar_predicciones(self, df_mensual, df_predicciones):
        """Crear visualización de ventas históricas y predichas"""
        plt.figure(figsize=(12, 6))
        
        # Graficar ventas históricas
        plt.plot(df_mensual.index, df_mensual['total_ventas_diarias'], 
                 label='Ventas Históricas', color='blue')
        
        # Graficar ventas predecidas
        plt.plot(df_predicciones.index, df_predicciones['ventas_predecidas'], 
                 label='Ventas Predecidas', color='red', linestyle='--')
        
        plt.title('Ventas Mensuales: Históricas y Predichas')
        plt.xlabel('Fecha')
        plt.ylabel('Total de Ventas')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt
    
    def ejecutar(self):
        """Ejecutor principal de la aplicación"""
        st.title('Panel de Predicción de Ventas')
        
        # Conectar a base de datos
        engine = self.conectar_base_datos()
        if not engine:
            return
        
        try:
            # Obtener y preparar datos
            datos_ventas = self.obtener_datos_ventas(engine)
            if datos_ventas is None:
                return
            
            datos_mensuales = self.preparar_datos_series_temporales(datos_ventas)
            if datos_mensuales is None:
                return
            
            # Entrenar modelo
            modelo, escalador = self.entrenar_modelo_prediccion_ventas(datos_mensuales)
            if modelo is None:
                return
            
            # Predecir próximos 4 meses
            predicciones = self.predecir_proximos_4_meses(datos_mensuales, modelo, escalador)
            if predicciones is None:
                return
            
            # Visualizaciones
            st.subheader('Predicción de Ventas para los Próximos 4 Meses')
            
            # Gráfico de predicciones
            grafico_ventas = self.visualizar_predicciones(datos_mensuales, predicciones)
            st.pyplot(grafico_ventas)
            
            # Tabla de predicciones
            st.subheader('Ventas Mensuales Predecidas')
            st.dataframe(predicciones)
            
            # Información adicional
            st.subheader('Información de Rendimiento del Modelo')
            st.write(f"Últimas Ventas Mensuales Conocidas: S/. {datos_mensuales['total_ventas_diarias'].iloc[-1]:,.2f}")
            st.write("Ventas Predecidas para los Próximos 4 Meses:")
            for fecha, ventas in predicciones.iterrows():
                st.write(f"{fecha.strftime('%B %Y')}: S/. {ventas['ventas_predecidas']:,.2f}")
        
        except Exception as e:
            self.logger.error(f"Error inesperado: {e}")
            st.error("Ocurrió un error inesperado en la aplicación")

if __name__ == "__main__":
    app = AplicacionPrediccionVentas()
    app.ejecutar()