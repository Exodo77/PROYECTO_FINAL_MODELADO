# Demo Streamlit app

Pequeña demo que permite introducir variables de usuario y obtener la probabilidad de compra estimada por el modelo.

Cómo ejecutar (desde la raíz del proyecto):

```powershell
python -m pip install -r requirements.txt
streamlit run apps/streamlit_app.py
```

Notas:
- Asegúrate de tener `models/modelo_ganador.pkl` y opcionalmente `models/preprocessor.pkl` en la carpeta `models/`.
- La app es demostrativa y usa un subconjunto reducido de features para mantener la UI didáctica.
