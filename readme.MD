# Demostración Redes Neuronales Convolucionales

En este proyecto se brinda un ejemplo de la forma de implementar una red neuronal convolucional capaz de detectar dígitos escritos a mano.

## Instalación

1. Clona este repositorio:

    ```bash
    git remote add origin https://github.com/felipebuitragocarmona/demo_cnn.git
    ```

2. Navega hasta el directorio del proyecto:

    ```bash
    cd demo_cnn
    ```
   
3. Crea y activa un entorno virtual:

    ```bash
    # Para crear el entorno virtual
    python -m venv venv  # Para Python 3.x
    # o
    virtualenv venv  # Para Python 2.x

    # Para activar el entorno virtual
    source venv/bin/activate  # Para Linux/Mac
    # o
    venv\Scripts\activate  # Para Windows

4. Instala las dependencias utilizando pip:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

Para entrenar el modelo se debe ejecutar el siguiente script

```python
python entrenamiento.py
```

Luego para realizar las pruebas se debe ejecutar 

```python
python PruebasCNN.py
```