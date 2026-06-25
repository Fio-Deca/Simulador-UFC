import pandas as pd
import re
def simular_pelea_final(peleador_A, peleador_B, df_peleadores, pipeline_modelo, columnas_entrenamiento):
    # 1. Buscar a los peleadores
    try:
        A = df_peleadores[df_peleadores['name'] == peleador_A].iloc[0]
        B = df_peleadores[df_peleadores['name'] == peleador_B].iloc[0]
    except IndexError:
        return "❌ Peleador no encontrado en la base de datos."

    # --- 🚨 CONTROL DE COMISIÓN ATLÉTICA ESTRICTO (KILOS A LIBRAS) 🚨 ---
    NOMBRE_COLUMNA_PESO = 'weight'

    if NOMBRE_COLUMNA_PESO not in df_peleadores.columns:
        print(f"🚫 SIMULACIÓN CANCELADA: No encuentro la columna '{NOMBRE_COLUMNA_PESO}'.")
        return

    try:
        # Extraemos los números (que en tu dataset están en kilos)
        peso_A_kg = float(re.sub(r'[^\d.]', '', str(A[NOMBRE_COLUMNA_PESO])))
        peso_B_kg = float(re.sub(r'[^\d.]', '', str(B[NOMBRE_COLUMNA_PESO])))

        # ¡LA MAGIA! Convertimos los kilos a libras para comparar con la UFC
        peso_A_lbs = peso_A_kg * 2.20462
        peso_B_lbs = peso_B_kg * 2.20462

        # Divisiones oficiales de la UFC en libras
        divisiones = [115, 125, 135, 145, 155, 170, 185, 205, 265]

        # Encontramos la división más cercana
        indice_A = min(range(len(divisiones)), key=lambda i: abs(divisiones[i] - peso_A_lbs))
        indice_B = min(range(len(divisiones)), key=lambda i: abs(divisiones[i] - peso_B_lbs))

        diferencia_saltos = abs(indice_A - indice_B)

        if diferencia_saltos > 1:
            # Creamos el texto de error
            error_peso = f"""
            ⚖️ COMBATE DENEGADO POR LA COMISIÓN ATLÉTICA ⚖️
            {"-" * 40}
            Diferencia ilegal de categorías detectada:
            🔴 {peleador_A} pertenece a {divisiones[indice_A]} lbs.
            🔵 {peleador_B} pertenece a {divisiones[indice_B]} lbs.
            => Solo se permite enfrentar a peleadores con máximo 1 salto de división.
            """
            return error_peso # 👈 Devolvemos el error a la web en lugar de un 'return' vacío

    except Exception as e:
        print(f"🚫 ERROR al leer los pesos: {e}")
        return
    # --------------------------------------------------------

    # 2. Calcular las diferencias (Solo las métricas de combate)
    datos_dict = {
        'ape_index_diff': A['ape_index'] - B['ape_index'],
        'bmi_diff': A['BMI'] - B['BMI'],
        'splm_diff': A['splm'] - B['splm'],
        'sapm_diff': A['sapm'] - B['sapm'],
        'str_acc_diff': (A['str_acc'] - B['str_acc']) / 100,
        'str_def_diff': (A['str_def'] - B['str_def']) / 100,
        'td_avg_diff': A['td_avg'] - B['td_avg'],
        'td_def_diff': A['td_def'] - B['td_def'],
        'sub_avg_diff': A['sub_avg'] - B['sub_avg'],
        'wins_diff': A['wins'] - B['wins'],
        'losses_diff': A['losses'] - B['losses']
    }

    # Creamos el DataFrame y rellenamos
    df_pred = pd.DataFrame([datos_dict])
    for col in columnas_entrenamiento:
        if col not in df_pred.columns:
            df_pred[col] = 0
    df_pred = df_pred[columnas_entrenamiento]

# 3. Predicción real con predict_proba
    probabilidades = pipeline_modelo.predict_proba(df_pred)[0]

    prob_A = probabilidades[1] * 100
    prob_B = probabilidades[0] * 100

    # --- 🚨 ESTO ES LO QUE DEBES CAMBIAR 🚨 ---
    # En lugar de hacer prints sueltos, guardamos todo el texto en una variable
    texto_resultado = f"""
    🥊 SIMULACIÓN DE COMBATE 🥊
    
    {peleador_A} vs {peleador_B}
    {"-" * 40}
    Probabilidad de {peleador_A}: {prob_A:.2f}%
    Probabilidad de {peleador_B}: {prob_B:.2f}%
    {"-" * 40}
    """
    
      # En lugar de un texto, devolvemos un diccionario con los datos puros
    return {
        "peleador_rojo": peleador_A,
        "peleador_azul": peleador_B,
        "prob_rojo": prob_A,
        "prob_azul": prob_B
    }
