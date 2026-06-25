import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from funciones import simular_pelea_final
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans # 👈 ¡AÑADE ESTA LÍNEA!

st.set_page_config(page_title="Simulador UFC", page_icon="🥊", layout="centered")

# 1. FUNCIÓN PARA CLASIFICAR LAS DIVISIONES AL VUELO
def obtener_nombre_division(peso_kg):
    peso_lbs = peso_kg * 2.20462
    limites = [115, 125, 135, 145, 155, 170, 185, 205, 265]
    nombres = [
        "Strawweight (115 lbs / 52 kg)",
        "Flyweight (125 lbs / 56 kg)",
        "Bantamweight (135 lbs / 61 kg)",
        "Featherweight (145 lbs / 65 kg)",
        "Lightweight (155 lbs / 70 kg)",
        "Welterweight (170 lbs / 77 kg)",
        "Middleweight (185 lbs / 83 kg)",
        "Light Heavyweight (205 lbs / 93 kg)",
        "Heavyweight (265 lbs / 120 kg)"
    ]
    idx = min(range(len(limites)), key=lambda i: abs(limites[i] - peso_lbs))
    return nombres[idx]

def limpiar_division_peleas(texto):
    texto = str(texto).lower()
    
    # 🚨 NUEVO: Descartamos Catchweight y Open Weight inmediatamente
    if 'catch weight' in texto or 'open weight' in texto: 
        return "Otros Torneos"
        
    if 'women' in texto:
        if 'strawweight' in texto: return "Women's Strawweight"
        if 'flyweight' in texto: return "Women's Flyweight"
        if 'bantamweight' in texto: return "Women's Bantamweight"
        if 'featherweight' in texto: return "Women's Featherweight"
        
    if 'light heavyweight' in texto: return "Light Heavyweight"
    if 'heavyweight' in texto: return "Heavyweight"
    if 'strawweight' in texto: return "Strawweight"
    if 'flyweight' in texto: return "Flyweight"
    if 'bantamweight' in texto: return "Bantamweight"
    if 'featherweight' in texto: return "Featherweight"
    if 'lightweight' in texto: return "Lightweight"
    if 'welterweight' in texto: return "Welterweight"
    if 'middleweight' in texto: return "Middleweight"
    
    return "Otros Torneos"
# 2. CARGA DE RECURSOS
@st.cache_resource
def load_resources():
    df = pd.read_csv('DATASET/fighter_details_final.csv')
    df['division'] = df['weight'].apply(obtener_nombre_division)
    
    df_peleas = pd.read_csv('DATASET/UFC.csv') 
    
    # Aplicamos la limpieza
    if 'division' in df_peleas.columns:
        df_peleas['weight_class_clean'] = df_peleas['division'].apply(limpiar_division_peleas)
    
    modelo = joblib.load('modelo_prediccion_ufc.joblib')
    cols = modelo.feature_names_in_ 
    return df, df_peleas, modelo, cols

df, df_peleas, modelo, cols = load_resources()

# --- INTERFAZ VISUAL PRINCIPAL ---
st.markdown("<h1 style='text-align: center;'>🥊 Simulador Predictivo UFC 🥊</h1>", unsafe_allow_html=True)
st.divider()

# Creamos dos pestañas en la parte superior
tab_simulador, tab_eda = st.tabs(["🎮 Simulador de Combates", "📊 Análisis Exploratorio (EDA)"])

# ==========================================
# 🥊 PESTAÑA 1: EL SIMULADOR
# ==========================================
with tab_simulador:
    st.subheader("📁 1. Elige la Categoría")
    
    # 🚨 CORRECCIÓN 1: Leemos de 'df', NO de 'df_peleas' para que salgan solo las 9 categorías limpias
    lista_divisiones = sorted(df['division'].dropna().unique())
    division_elegida = st.selectbox("Categorías disponibles:", lista_divisiones)

    st.divider()

    df_filtrado = df[df['division'] == division_elegida]
    lista_peleadores_filtrados = sorted(df_filtrado['name'].unique())

    st.subheader("🤼 2. Selecciona a los Contendientes")
    col1, col_vs, col2 = st.columns([4, 1, 4])

    with col1:
        st.markdown("**🔴 Esquina Roja**")
        p1 = st.selectbox("Peleador Rojo", lista_peleadores_filtrados, key="p1", label_visibility="collapsed")

    with col_vs:
        st.markdown("<h3 style='text-align: center; margin-top: 10px;'>VS</h3>", unsafe_allow_html=True)

    with col2:
        st.markdown("**🔵 Esquina Azul**")
        p2 = st.selectbox("Peleador Azul", lista_peleadores_filtrados, key="p2", label_visibility="collapsed")

    st.write("")

    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        boton_simulado = st.button("🔥 ¡Simular Combate!", use_container_width=True)

    if boton_simulado:
        st.divider()
        resultado = simular_pelea_final(p1, p2, df, modelo, cols)
        
        if isinstance(resultado, str):
            st.error(resultado, icon="⚖️")
        else:
            st.subheader("🏆 Predicción del Modelo")
            rojo, azul = resultado["peleador_rojo"], resultado["peleador_azul"]
            p_rojo, p_azul = resultado["prob_rojo"], resultado["prob_azul"]
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric(label=f"🔴 {rojo}", value=f"{p_rojo:.2f}%")
                st.progress(int(p_rojo))
            with col_res2:
                st.metric(label=f"🔵 {azul}", value=f"{p_azul:.2f}%")
                st.progress(int(p_azul))
                
            st.write("")
            ganador = rojo if p_rojo > p_azul else azul
            st.success(f"**Ganador Más Probable:** {ganador} 🥊")
            st.balloons()


# ==========================================
# 📊 PESTAÑA 2: ANÁLISIS EXPLORATORIO (EDA)
# ==========================================
with tab_eda:
# 2. GRÁFICO: Cómo finalizan los combates (En Porcentajes)
    # 2. GRÁFICO: Cómo finalizan los combates (Con etiquetas de %)
    st.subheader("📊 Métodos de Finalización por División (%)")
    st.write("Análisis histórico de la probabilidad de finalización en cada categoría de peso.")
    
    col_division = 'weight_class_clean' if 'weight_class_clean' in df_peleas.columns else 'division'
    col_metodo = 'method'         
    
    if col_division in df_peleas.columns and col_metodo in df_peleas.columns:
        df_grafico_peleas = df_peleas[df_peleas[col_division] != "Otros Torneos"].copy()
        
        orden_oficial_ufc = [
            "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
            "Flyweight", "Bantamweight", "Featherweight",
            "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"
        ]
        
        df_grafico_peleas[col_division] = pd.Categorical(df_grafico_peleas[col_division], categories=orden_oficial_ufc, ordered=True)
        
        # Agrupamos y calculamos porcentajes
        df_conteos = df_grafico_peleas.groupby([col_division, col_metodo], observed=False).size().unstack(fill_value=0)
        df_conteos = df_conteos.dropna(how='all')
        df_porcentajes = df_conteos.div(df_conteos.sum(axis=1), axis=0) * 100
        
        # --- 🚀 NUEVO: MAGIA DE PLOTLY ---
        # Convertimos la tabla al formato "largo" que necesita Plotly
        df_grafico = df_porcentajes.reset_index().melt(id_vars=col_division, var_name='Método', value_name='Porcentaje')
        
        # Creamos el gráfico de barras apiladas
        fig = px.bar(
            df_grafico,
            x=col_division,
            y='Porcentaje',
            color='Método',
            text=df_grafico['Porcentaje'].apply(lambda x: f'{x:.1f}%' if x > 4 else ''), # Oculta textos si la franja es muy estrecha (<4%)
            color_discrete_sequence=px.colors.qualitative.Pastel # Paleta de colores elegante
        )
        
        # Ajustamos el diseño visual
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Proporción (%)",
            legend_title="Método de Victoria",
            xaxis={'categoryorder':'array', 'categoryarray':orden_oficial_ufc}, # Forza el orden en el eje X
            margin=dict(l=0, r=0, t=30, b=0)
        )
        # Posicionamos el texto dentro de las barras en negrita
        fig.update_traces(textposition='inside', textfont=dict(size=12, color='white', family="Arial Black"))
        
        # Enviamos el gráfico a Streamlit
        st.plotly_chart(fig, use_container_width=True)
        # ---------------------------------
        
        st.info("💡 **Lectura del gráfico:** Las barras muestran la proporción de resultados ordenados oficialmente de femenino a masculino y de menor a mayor pesaje.")
        
        # Opcional: Mostrar la tabla exacta de porcentajes debajo por si el tribunal quiere ver los decimales
        with st.expander("Ver tabla exacta de porcentajes"):
            st.dataframe(df_porcentajes.style.format("{:.2f}%"))
            
    else:
        st.warning(f"⚠️ No se encontró la columna '{col_division}' o '{col_metodo}' en df_peleas.")
    # 3. GRÁFICO: PCA + CLUSTERING DE ESTILOS DE PELEA
    st.subheader("🧬 Perfilado de Estilos de Pelea (PCA + K-Means)")
    st.write("Agrupación no supervisada de peleadores basada exclusivamente en su ADN de combate (estadísticas de *Striking* y *Grappling*).")
    
    try:
        # 1. Seleccionamos las columnas que definen el estilo de un peleador
        columnas_estilo = ['splm', 'sapm', 'str_acc', 'str_def', 'td_avg', 'td_def', 'sub_avg']
        
        # Comprobamos cuáles de estas columnas existen realmente en tu dataset
        cols_disponibles = [col for col in columnas_estilo if col in df.columns]
        
        # Limpiamos los peleadores que no tengan estos datos registrados
        df_pca_clean = df.dropna(subset=cols_disponibles).copy()
        
        if not df_pca_clean.empty and len(cols_disponibles) >= 2:
            X_estilo = df_pca_clean[cols_disponibles]
            
            # 2. Estandarizamos (vital para que un derribo valga lo mismo que un golpe a nivel matemático)
            X_scaled = StandardScaler().fit_transform(X_estilo)
            
            # 3. PCA para reducir a 2 dimensiones (para poder dibujarlo en pantalla)
            pca_engine = PCA(n_components=2)
            coordenadas_pca = pca_engine.fit_transform(X_scaled)
            
            # 4. K-Means para encontrar 3 perfiles de pelea automáticamente
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df_pca_clean['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # --- 🤖 LÓGICA INTELIGENTE PARA NOMBRAR LOS ESTILOS ---
            # Calculamos las medias de cada grupo para saber en qué destacan
            centroides = df_pca_clean.groupby('Cluster')[cols_disponibles].mean()
            
            # Asumimos que el grupo con más derribos (td_avg) son los Grapplers
            grappler_id = centroides['td_avg'].idxmax() if 'td_avg' in cols_disponibles else 0
            
            # Asumimos que el grupo con más golpes por minuto (splm) son los Strikers
            striker_id = centroides['splm'].idxmax() if 'splm' in cols_disponibles else 1
            
            # Si por alguna casualidad un grupo domina ambas, usamos nombres genéricos
            if grappler_id == striker_id:
                mapa_estilos = {0: "Estilo A", 1: "Estilo B", 2: "Estilo C"}
            else:
                # El grupo restante es el equilibrado
                mixto_id = [i for i in [0, 1, 2] if i not in [grappler_id, striker_id]][0]
                mapa_estilos = {
                    grappler_id: "🤼 Grapplers (Lucha de Suelo)", 
                    striker_id: "🥊 Strikers (Golpeo de Pie)", 
                    mixto_id: "⚖️ Estilo Mixto / Equilibrado"
                }
            
            # Asignamos el nombre final a cada peleador
            df_pca_clean['Perfil de Pelea'] = df_pca_clean['Cluster'].map(mapa_estilos)
            
            # 5. Preparamos los datos para Streamlit
            df_grafico_pca = pd.DataFrame(coordenadas_pca, columns=['Eje PCA 1', 'Eje PCA 2'])
            df_grafico_pca['Perfil de Pelea'] = df_pca_clean['Perfil de Pelea'].values
            
            st.write(f"Varianza explicada por el modelo 2D: **{sum(pca_engine.explained_variance_ratio_)*100:.2f}%**")
            
            # 6. Dibujamos el gráfico
            st.scatter_chart(
                data=df_grafico_pca, 
                x='Eje PCA 1', 
                y='Eje PCA 2', 
                color='Perfil de Pelea'
            )
            
            st.info("💡 **Nota metodológica:** Cada punto es un peleador. El algoritmo ha segmentado a los atletas en 3 perfiles analizando únicamente sus métricas numéricas, sin conocer sus nombres, métodos de victoria ni divisiones.")
        else:
            st.warning("Faltan métricas suficientes de striking/grappling en el dataset para realizar el perfilado.")
    except Exception as e:
        st.error(f"Error al calcular el PCA de estilos: {e}") 

    # 4. Mostrar el DataFrame (Tabla interactiva)
    st.subheader("Vista Previa del Dataset de Peleadores")
    st.dataframe(df, use_container_width=True)
