from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
import numpy as np

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo_entrenado_insectos.pkl')
app.logger.debug('Modelo cargado correctamente.')


mean_values = np.array([4.4, 5.17])  
std_values = np.array([3.05, 2.65]) 

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        abdomen = float(request.form['abdomen'])
        antenna = float(request.form['antenna'])

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[abdomen, antenna]], columns=['abdomen', 'antenna'])
        app.logger.debug(f'DataFrame creado: {data_df}')


        data_array = data_df.values
        data_scaled = (data_array - mean_values) / std_values
        app.logger.debug(f'Datos normalizados: {data_scaled}')

        # Realizar predicciones
        prediction = model.predict(data_scaled)
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoría': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)