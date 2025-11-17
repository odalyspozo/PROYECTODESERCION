# ============================================
# Instalar dependencias antes de ejecutar:
# pip install streamlit pandas scikit-learn plotly seaborn openpyxl
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================
# Título de la app
# ============================================
st.title("📊 Sistema de Alertas Tempranas para Retención Estudiantil")

# ============================================
# Subir archivo
# ============================================
uploaded_file = st.file_uploader("Carga el archivo Excel del dataset", type=["xlsx"])

if uploaded_file is None:
    st.info("Por favor, carga un archivo para comenzar.")
else:
    # Leer datos
    df = pd.read_excel(uploaded_file, sheet_name="AsignaturasEstudiante", engine="openpyxl")
    st.success("✅ Archivo cargado correctamente")

    # ============================================
    # Limpieza de datos - GUARDAR CÉDULA PARA USO POSTERIOR
    # ============================================
    # Guardar la cédula antes de eliminarla para usarla después
    cedulas = df["cedula_alumno"].copy()
    
    cols_to_drop = ["cedula_alumno", "apellido", "nombre", "celular", "email", "email_institucional",
                    "codigo_asignatura", "asignatura"]
    df_clean = df.drop(columns=cols_to_drop)

    nota_cols = ["parcial1", "parcial2", "parcial3", "sumatoriaparcial4", "examen_final", "recuperacion", "NOTA FINAL"]
    df_clean[nota_cols] = df_clean[nota_cols].fillna(0)
    df_clean['aprobado'] = df_clean['aprobado'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)

    st.subheader("Vista previa del dataset")
    st.dataframe(df_clean.head())

    # ============================================
    # Análisis Exploratorio
    # ============================================
    st.subheader("📈 Análisis Exploratorio")
    fig1 = px.histogram(df_clean, x="carrera", title="Distribución de Estudiantes por Carrera")
    st.plotly_chart(fig1)

    aprobacion_por_carrera = df_clean.groupby("carrera")["aprobado"].mean().reset_index()
    fig2 = px.bar(aprobacion_por_carrera, x="carrera", y="aprobado",
                  title="Tasa de Aprobación por Carrera", labels={"aprobado": "Tasa de Aprobación"})
    st.plotly_chart(fig2)

    st.write("Mapa de calor de correlaciones:")
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_clean[nota_cols + ["aprobado"]].corr(), annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig_corr)

    # ============================================
    # Preparación para el modelo
    # ============================================
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(columns=["aprobado"])
    y = df_encoded["aprobado"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # ============================================
    # Entrenar modelo
    # ============================================
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("📌 Resultados del Modelo")
    st.write(f"**Exactitud del modelo:** {accuracy:.2f}")
    st.text("Reporte de clasificación:")
    st.text(classification_report(y_test, y_pred))

    # ============================================
    # Alertas tempranas
    # ============================================
    probabilities = model.predict_proba(X_test)[:, 1]
    alert_threshold = st.slider("Umbral de alerta (probabilidad mínima de aprobar)", 0.0, 1.0, 0.5)

    alerts = X_test.copy()
    alerts['probabilidad_aprobado'] = probabilities
    alerts['alerta_desercion'] = alerts['probabilidad_aprobado'] < alert_threshold

    # Recuperar información original incluyendo la cédula
    alerts_full = df.iloc[alerts.index].copy()
    alerts_full['probabilidad_aprobado'] = probabilities
    alerts_full['alerta_desercion'] = alerts['alerta_desercion']

    # ============================================
    # Filtro por carrera
    # ============================================
    st.subheader("⚠️ Estudiantes en Riesgo por Carrera")
    carreras = alerts_full['carrera'].unique()
    carrera_seleccionada = st.selectbox("Selecciona una carrera", carreras)

    riesgo_carrera = alerts_full[(alerts_full['carrera'] == carrera_seleccionada) & (alerts_full['alerta_desercion'])]
    st.write(f"Estudiantes en riesgo en **{carrera_seleccionada}**:")
    
    # Mostrar las columnas disponibles (incluyendo cédula que conservamos del dataframe original)
    st.dataframe(riesgo_carrera[['cedula_alumno', 'apellido', 'nombre','carrera', 'NOTA FINAL', 'probabilidad_aprobado']])

    # Botón para descargar alertas
    csv = alerts_full.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar todas las alertas en CSV", csv, "alertas_retencion.csv", "text/csv")

    # Visualización de probabilidades
    fig3 = px.histogram(probabilities, nbins=20,
                        title="Distribución de Probabilidad de Aprobación",
                        labels={'value': 'Probabilidad', 'count': 'Cantidad'})
    st.plotly_chart(fig3)