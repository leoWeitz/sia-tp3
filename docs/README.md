# TP3 SIA — Perceptrón Simple y Multicapa · Specs SDD

**Sistemas de Inteligencia Artificial — ITBA — 2º cuatrimestre 2026**

| Integrante | Legajo |
|---|---|
| Celestino Garrós | 64375 |
| Leo Weitz | 64365 |
| Federico Ignacio Ruckauf | 64356 |
| Matías Romanato | 62072 |

Esta carpeta es la **fuente de verdad** para lo que falta implementar. El motor existente (Etapas 0–5 de `PLAN_MOTOR.md`) se respeta tal cual: las specs están escritas **sobre sus interfaces reales**. Si código y spec no coinciden, se corrige uno de los dos explícitamente, acordándolo con el equipo.

## Archivos

| Archivo | Para qué sirve |
|---|---|
| [`01-constitucion.md`](01-constitucion.md) | Reglas del proyecto (complementa `CLAUDE.md` §0 y §2) |
| [`02-requerimientos.md`](02-requerimientos.md) | Requerimientos del enunciado y trazabilidad pregunta → experimento → figura → slide |
| [`03-arquitectura.md`](03-arquitectura.md) | Interfaces reales del motor, extensiones previstas, contratos de config y resultados |
| [`04-matematica.md`](04-matematica.md) | Fórmulas con la notación de la cátedra y su equivalencia con las convenciones del código |
| [`PLAN_MOTOR.md`](PLAN_MOTOR.md) | Plan original del motor. Queda como **histórico**: sus Etapas 0–5 están hechas y sus Etapas 6–8 las reemplazan F04–F07 y F14 |
| [`verificacion_manual.md`](verificacion_manual.md) | Iteración de backprop hecha a mano en `[2, 2, 1]`, reproducida por un test |
| [`verificacion_manual_2321.md`](verificacion_manual_2321.md) | Ídem en `[2, 3, 2, 1]` (dos capas ocultas) |
| `fases/` | Una spec por fase |
| `datos/` | Diccionarios de datos y decisiones sobre los datos (se completan en las EDA) |
| `resultados/` | Respuestas a las preguntas del enunciado (se completan con los experimentos) |

## Fases y estado

| Fase | Spec | Depende de | Estado |
|---|---|---|---|
| F00 | [Setup](fases/F00-setup.md) | — | ✅ Hecho |
| F01 | [Núcleo numérico](fases/F01-nucleo.md) | — | ✅ Hecho (β de la logística alineado con la cátedra en F04-T0) |
| F02 | [Perceptrón simple](fases/F02-perceptron-simple.md) | — | ✅ Hecho |
| F03 | [Multicapa + backprop](fases/F03-mlp.md) | — | ✅ Hecho |
| F04 | [Optimizadores, regularización y extensiones de `fit`](fases/F04-optimizadores.md) | motor | ✅ Hecho |
| F05 | [Datos](fases/F05-datos.md) | — (conviene después de F09) | ✅ Hecho |
| F06 | [Métricas y umbral](fases/F06-metricas.md) | — | ✅ Hecho |
| F07 | [Runner, configs, save/load](fases/F07-infra-experimentos.md) | F04, F05, F06 | ✅ Hecho |
| F08 | [Validación](fases/F08-validacion.md) | F07 (solo figuras) | ✅ Hecho · falta T3 (verificación en papel) |
| F09 | [Ej1 · EDA fraude](fases/F09-ej1-eda.md) | — (solo pandas) | ⏳ |
| F10 | [Ej1 · Aprendizaje](fases/F10-ej1-aprendizaje.md) | F07, F09 | ⏳ |
| F11 | [Ej1 · Generalización y umbral](fases/F11-ej1-generalizacion.md) | F10 | ⏳ |
| F12 | [Ej2 · Dígitos](fases/F12-ej2-digitos.md) | Parte 1 (EDA): — · resto: F07 | ⏳ |
| F13 | [Ej3 · More digits ≥ 98 %](fases/F13-ej3-more-digits.md) | F12 | ⏳ |
| F14 | [Análisis y figuras](fases/F14-analisis.md) | F10–F13 | ⏳ |
| F15 | [Presentación](fases/F15-presentacion.md) | F14 | ⏳ |
| F16 | [Opcionales](fases/F16-opcionales.md) | F15 en borrador | ⏳ |

**Qué se puede hacer en paralelo:**
- Ya mismo, sin depender entre sí: **F04, F05, F06, F09 y la Parte 1 (EDA) de F12**. Tocan archivos distintos (`03-arquitectura.md` §1).
- **F07** necesita F04, F05 y F06 (puede arrancar con datasets sintéticos y conectar `data/` y `core/metrics.py` cuando estén mergeados).
- Después de F07, **Ej1 (F10 → F11)** y **Ej2/Ej3 (F12 → F13)** son independientes entre sí.
- **Camino crítico:** F04 → F07 → F12 → F13 (los barridos de dígitos son lo que más tiempo de cómputo lleva).

## Flujo SDD con el agente

1. Una fase por sesión del agente, en su propia rama (`f05-datos`).
2. En el prompt se le pide que lea primero `CLAUDE.md`, `01-constitucion.md`, `03-arquitectura.md`, `04-matematica.md` y la spec de la fase.
3. Plan corto → OK humano → tests primero → código → `pytest` + `ruff` en verde.
4. El agente muestra los cambios y propone los mensajes de commit. **Commit, push y PR los hace una persona** (o el agente, solo con permiso explícito y sin co-autoría: `CLAUDE.md` §0).
5. PR a `main` con CI en verde, revisado por otra persona del equipo.
6. Al cerrar la fase: tildar los criterios de aceptación en la spec y actualizar la columna Estado de esta tabla.