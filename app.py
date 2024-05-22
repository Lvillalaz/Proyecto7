from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Cargar el modelo entrenado al iniciar la aplicación
model = joblib.load('modelo_calificacion.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos de la solicitud POST
        data = request.get_json(force=True)

        # Verificar que los datos esperados estén presentes y tengan el formato adecuado
        required_fields = ["Rating", "Reviews", "Installs", "Price"]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Convertir los datos a float y realizar la predicción con el modelo cargado
        input_data = [float(data[field]) for field in required_fields]
        prediction = model.predict([input_data])

        # Devolver el resultado como JSON
        return jsonify({'prediction': prediction.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)