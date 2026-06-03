# Cómo ejecutar el código

Si `uv` no está instalado, checkea [las siguientes instrucciones de instalación](https://docs.astral.sh/uv/getting-started/installation/).

Para instalar las dependencias del proyecto, ejecutar en la raíz del repositorio el siguiente comando:

```
uv sync --dev
```

Esto va a crear el entorno virtual `dinamica-adaptativa`, el cuál debe utilizarse para revisar tipado estático de los scripts de Python, y para ejecutar los jupyter notebooks.

Alternativamente, es posible usar entornos globales como Anaconda/Miniconda, pero el entorno de ejecución junto a sus dependencias se tendrán que instalar manualmente.

# Planeación

- [x] Resolver esquema general unidimensional.
- [ ] Crear funciones para plotear las funciones del modelo y la solución. ***[en progreso]***
- [ ] Estudiar monomorficidad y dimorficidad. ***[en progreso]***
- [x] Probar distintos métodos de diferencias finitas (explícito, implícito y semi-implícito).
- [ ] Estudiar oscilaciones y sus posibles orígenes (en caso de aparecer).
- [ ] Probar métodos espectrales. ***[en progreso]***
- [ ] Simular canibalismo.
- [ ] Replicar los resultados del paper en el caso gaussiano. ***[en progreso]***
- [ ] Ver qué pasa al cambiar condiciones en tiempos arbitrarios.
- [ ] Resolver esquema general bidimensional. (propuesto)
- [ ] Estudiar polimorficidad. (propuesto)


