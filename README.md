# Cómo ejecutar el código

Si `uv` no está instalado, checkea [las siguientes instrucciones de instalación](https://docs.astral.sh/uv/getting-started/installation/).

Para instalar las dependencias del proyecto, ejecutar en la raíz del repositorio el siguiente comando:

```
uv sync --dev
```

Esto va a crear el entorno virtual `dinamica-adaptativa`, el cuál debe utilizarse para revisar tipado estático de los scripts de Python, y para ejecutar los jupyter notebooks.

# Planeación

- [x] Resolver esquema general unidimensional.
- [ ] Crear funciones para plotear las funciones del modelo y la solución.
- [ ] Estudiar monomorficidad y dimorficidad.
- [x] Probar distintos métodos de diferencias finitas (explícito, implícito y semi-implícito).
- [ ] Ver qué pasa al cambiar condiciones en tiempos arbitrarios.
- [ ] Resolver esquema general bidimensional.
- [ ] Estudiar polimorficidad.

# Consideraciones

- [x] Simular para el caso gaussiano.
- [ ] Simular canibalismo.
- [ ] Encontrar CFL.
- [ ] Estudiar oscilaciones y sus posibles orígenes.
