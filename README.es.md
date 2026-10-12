[ 🇺🇸 [Read in English](README.md) ] | [ 🇨🇱 Español ]

# Recomendador de e-commerce sobre 85 millones de eventos reales

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![DuckDB](https://img.shields.io/badge/duckdb-SQL-FFF000?logo=duckdb&logoColor=black)
![LightGBM](https://img.shields.io/badge/lightgbm-lambdarank-9ACD32)
![Tests](https://github.com/Rxyxs/ecommerce-recommender-rees46/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

Un recomendador en dos etapas (generación de candidatos y luego un modelo learning-to-rank) entrenado y evaluado offline sobre dos meses de clickstream real de una tienda online multicategoría. Para 50.000 compradores de la última semana de noviembre, **el ranker LightGBM llega a un recall@20 de 0,395 contra 0,244 del mejor generador solo**, "lo que el usuario vio hace poco". El techo está fijado antes de rankear: **el 71,4% de lo que se compra esa semana es un producto que el usuario nunca había visto**, y solo el 56,0% de lo comprado aparece entre los candidatos de algún generador.

## Datos

[eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) (Kaggle, REES46 / Open CDP), octubre–noviembre 2019, descargado el 2026-10-11. **Licencia "copyright-authors"**: los datos no están en este repositorio ni se redistribuyen; `python -m src.data` los descarga con tu propio token de Kaggle, y solo se publican métricas agregadas.

El proyecto hermano [ecommerce-lakehouse-duckdb](https://github.com/Rxyxs/ecommerce-lakehouse-duckdb) auditó este dataset, y dos de sus hallazgos deciden qué se usa aquí:

- **Los eventos de carrito quedan fuera.** Su tracking cambia seis veces en dos meses, así que un modelo entrenado con carritos aprendería el tracking. Solo se usan vistas y compras: 85.290.907 eventos, 1.384.137 de ellos compras.
- **Del 14 al 17 de noviembre queda fuera.** En esos días no se registraron compras durante 48 horas de alto tráfico, así que ninguna ventana se evalúa con etiquetas faltantes.

## Diseño

Todos los cortes son por tiempo, nunca al azar:

| Ventana | Fechas | Rol |
|---|---|---|
| historial | 1 oct – 13 nov | candidatos y features para las etiquetas de entrenamiento del ranker |
| validación | 18 – 23 nov | etiquetas con que se entrena el ranker |
| test | 24 – 30 nov | solo evaluación; su historial llega al 23 de noviembre |

La tarea: para cada usuario que compra en la semana de test, ordenar 20 productos de modo que los que compra queden primero. 105.026 usuarios compran esa semana; las métricas promedian sobre una muestra aleatoria fija de 50.000 de los que tienen historial (semilla 42), y todos los métodos se evalúan sobre los mismos usuarios. Un usuario sin recomendación de un método cuenta como cero.

**Candidatos** (hasta 50 por generador, 152 por usuario al unirlos):

- `recent`: productos que el usuario vio en los últimos 30 días, del más reciente al más antiguo.
- `coview`: vecinos de los últimos cinco productos del usuario en un grafo de co-vistas por sesión de los últimos 14 días, normalizado por la popularidad del vecino.
- `als`: ALS implícito (64 factores) sobre 30 días de vistas (peso 1) y compras (peso 5).
- `popular`: los más vendidos de la última semana, en total y en la categoría principal del usuario.

**Ranker**: LightGBM `lambdarank` con 21 features: el ranking y el puntaje de cada generador, el historial usuario–producto (vistas, días desde la última vista, si ya lo compró, si estaba en su última sesión), la demanda del producto (vistas, compras y conversión), el precio respecto del precio habitual del usuario y su actividad.

## Resultados

![recall@20 y NDCG@20 por método](outputs/figures/methods.png)

| Método | recall@20 | hit@20 | NDCG@20 |
|---|---:|---:|---:|
| popular | 0,211 | 0,255 | 0,104 |
| ALS | 0,202 | 0,243 | 0,128 |
| co-vistos | 0,207 | 0,247 | 0,096 |
| vistos hace poco | 0,244 | 0,293 | 0,174 |
| **ranker LightGBM** | 0,395 | 0,453 | 0,255 |

Respecto del mejor generador solo, el ranker sube el recall@20 en 62% y el NDCG@20 en 47%. Cada generador capta una parte de la señal; el trabajo del ranker es decidir, para cada usuario, a cuál creerle, y las features en que se apoya muestran cómo:

![Importancia de features](outputs/figures/importance.png)

Cuántas veces el usuario vio el producto (32,2% de la ganancia) y cuánto se vendió la semana pasada (14,3%) explican casi la mitad del modelo. La gente compra lo que miró varias veces y lo que se está vendiendo ahora. El puntaje de ALS, la señal más compleja, aporta el 2,4%.

## El techo: la mayoría de las compras son de productos nunca vistos

![Recall de candidatos por generador](outputs/figures/candidate_recall.png)

El recall de candidatos es la proporción de pares (usuario, producto) comprados que aparecen en algún lugar entre los candidatos: ningún ranker puede recomendar lo que no está ahí. Los cuatro generadores juntos llegan al **56,0%**. La razón es el 71,4% de arriba: la mayoría de los productos comprados en la semana de test no tenían ninguna vista ni compra de ese usuario antes de armar la lista, así que ningún generador basado en historial podía proponerlos. Mejorar eso requiere señales de la sesión en curso (lo que el usuario está viendo ahora), no un mejor modelo del pasado.

## Usuarios sin historial

El 27,0% de los compradores de la semana de test no tenía ningún evento antes del 23 de noviembre. Para ellos la única opción offline son los más vendidos de la semana anterior, que dan un recall@20 de 0,277 sobre 28.343 usuarios: más que la misma lista para usuarios con historial (0,211), porque la primera compra de un comprador nuevo es más a menudo el producto más vendido de la tienda. Un sistema real cambiaría a recomendaciones por sesión después de la primera vista; con lotes diarios, este diseño offline no puede medirlo.

## Lo que las métricas offline no dicen

El recall@20 mide si los productos que un usuario compraría de todos modos están en la lista. No mide el ingreso **incremental**: si mostrar la lista hace que alguien compre algo que de otro modo no compraría. Para eso hace falta un test A/B, que los logs offline no reemplazan; este repositorio no afirma nada sobre ingresos.

## Reproducir

```bash
python -m venv .venv
.venv\Scripts\activate         # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m src.data              # descarga de Kaggle (~15 GB de CSV) -> data/events.parquet
python -m src.main              # ambas etapas, unos 3 minutos cada una en un PC de 8 núcleos -> reports/results.json
python -m src.make_figures
pytest                          # métricas y generadores sobre un fixture hecho a mano
```

`tests/test_readme.py` verifica cada cifra de este README contra `reports/results.json`.

## Limitaciones

- El ranker se entrena con etiquetas que empiezan cuatro días después del fin de su historial (el hueco del 14 al 17 de noviembre), mientras que las etiquetas de test empiezan al día siguiente. Ese desajuste juega en contra del ranker, así que, si algo, la ventaja sobre los baselines está subestimada.
- Una sola semana de test, que incluye el Black Friday (29 de noviembre). Los resultados en una semana más tranquila pueden diferir.
- Se evalúan 50.000 de los 105.026 compradores, para que la tabla de features quepa en memoria.
- ALS y el ranker usan una semilla cada uno; sin ajuste más allá de valores cercanos a los estándar.
