# 04 · Referencia matemática

Notación de la cátedra: $\xi^\mu$ (o $x^\mu$) entrada del dato $\mu$, $\zeta^\mu$ salida esperada, $O^\mu$ salida obtenida, $h$ excitación, $\theta$ activación, $\eta$ tasa de aprendizaje, $p$ cantidad de datos, $V^m$ salida de la capa $m$ (con $V^0 = \xi$). En el código (motor existente): `X` entradas, `y` esperadas, `y_pred` salidas, `z` pre-activación (= $h$), `a` activación (= $V$), `W (n_in, n_out)`, `b (1, n_out)`.

> **Convenciones del código que difieren de la teórica** (todas equivalentes, documentadas en §2.2, §2.4 y §3.3): la MSE es un **promedio sin ½**; `W` se guarda como **(n_in, n_out)**, o sea transpuesta respecto de $W_{ij}$ de la cátedra (i = neurona destino); la logística del código hoy usa β sin el 2 (se alinea en F04-T0).

Todo módulo que implemente una fórmula de este archivo debe citar la sección en su docstring.

---

## §1 Perceptrón simple

Excitación: $h^\mu = \sum_{i=1}^{n} w_i\,x_i^\mu + w_0 = w\cdot x^\mu + b$. Salida: $O^\mu = \theta(h^\mu)$.

### §1.1 Escalón (Rosenblatt) — Clase 10.1

$$\theta(h) = \begin{cases} 1 & h \ge 0\\ -1 & h < 0\end{cases}\qquad \Delta w = \eta\,(\zeta^\mu - O^\mu)\,x^\mu,\quad \Delta b = \eta\,(\zeta^\mu - O^\mu)$$

- Equivale a $\Delta w = 2\eta\,\zeta^\mu x^\mu$ cuando $O^\mu \ne \zeta^\mu$ y 0 en otro caso.
- En el código: `core/perceptron.py::fit_perceptron`, por lotes $\Delta W = \eta\,X^\top(y - \hat y)$, **sin** dividir por $n$ (con `batch_size=1` es la regla clásica). Ya implementado y testeado.
- Criterio de corte: error de clasificación $= \#\{\mu : O^\mu \ne \zeta^\mu\} = 0$ (accuracy 100 %) o máximo de épocas.

### §1.2 Lineal (ADALINE, Widrow–Hoff) — Clase 10.2

$$\theta(h) = h,\qquad E(w) = \tfrac12\sum_{\mu}(\zeta^\mu - O^\mu)^2,\qquad \Delta w = -\eta\frac{\partial E}{\partial w} = \eta\,(\zeta^\mu - O^\mu)\,x^\mu$$

### §1.3 No lineal — Clase 10.2

Misma función de error; $\Delta w = \eta\,(\zeta^\mu - O^\mu)\,\theta'(h^\mu)\,x^\mu$.

### §1.4 Modos de entrenamiento — Clase 11

| Modo | Actualización | En el código |
|---|---|---|
| Online / incremental (SGD) | después de cada dato | `batch_size=1`, datos barajados cada época |
| Minibatch (SGD) | después de un subconjunto | `batch_size=k`, barajado cada época |
| Batch (GD) | después de todos los datos | `batch_size=None` (lote completo) |

---

## §2 Activaciones y funciones de costo

### §2.1 Tangente hiperbólica
$\theta(h) = \tanh(\beta h)$, imagen $(-1,1)$, $\theta'(h) = \beta\,(1 - \theta^2(h))$.

### §2.2 Logística (convención de la cátedra, con 2β)
$\theta(h) = \dfrac{1}{1 + e^{-2\beta h}}$, imagen $(0,1)$, $\theta'(h) = 2\beta\,\theta(h)\,(1 - \theta(h))$.

**En el código hoy:** `Sigmoid(beta)` calcula $1/(1+e^{-\beta z})$, o sea `Sigmoid(beta=b)` = logística de la cátedra con $\beta = b/2$. Propuesta a confirmar con el equipo: alinear el código a la cátedra (F04-T0: cambiar a $2\beta$ en `forward` y `backward`), para que los β que se reportan en la presentación sean los de la teoría. Identidad útil para el test: $\theta_{2\beta}(h) = \tfrac12\big(1 + \tanh(\beta h)\big)$.

### §2.3 Extras (ya implementadas; se usan como variantes)
- ReLU: $\theta(h) = \max(0, h)$, $\theta'(h) = \mathbb{1}[h>0]$.
- Softmax (solo salida, solo con entropía cruzada): $O_i = e^{h_i - \max_j h_j} / \sum_k e^{h_k - \max_j h_j}$.

### §2.4 MSE (default)
Teórica: $E = \tfrac12\sum_\mu\sum_i (\zeta_i^\mu - O_i^\mu)^2$.

**En el código (`core/losses.py::MSE`, decisión 1 del `README.md` del repo):** promedio sobre las $n$ muestras del lote **y** las $m$ salidas, sin ½:
$$L = \frac{1}{n\,m}\sum_{\mu}\sum_i(\zeta_i^\mu - O_i^\mu)^2,\qquad \frac{\partial L}{\partial O} = \frac{2\,(O - \zeta)}{n\,m}$$
Relación: $L = \frac{2}{n\,m}E$. Un paso con $\eta_{\text{código}}$ equivale a un paso de la cátedra con $\eta = \frac{2\,\eta_{\text{código}}}{n\,m}$ (online con una salida: $\eta = 2\,\eta_{\text{código}}$). Ventaja: el valor de $L$ y la escala de η no dependen del tamaño de lote. Se aclara en la presentación cuando se muestren valores de η. En dígitos ($m = 10$) el η "efectivo por salida" queda dividido por 10 respecto de una MSE sumada sobre salidas: tenerlo en cuenta al elegir el rango del barrido de η.

### §2.5 Entropía cruzada (opcional, ya implementada)
- `BinaryCrossEntropy`: $L = -\frac{1}{nm}\sum[y\log\hat y + (1-y)\log(1-\hat y)]$, pensada para salida logística y target 0/1 (o probabilidad de BigModel: la BCE acepta soft labels).
- `CategoricalCrossEntropy`: suma sobre clases, promedia sobre muestras; con softmax la red usa `softmax_delta` $= (\hat y - y)/n$ directamente (decisión 2 del `README.md` del repo).

---

## §3 Perceptrón multicapa — Clase 11

### §3.1 Feed-forward (convención del código: `core/layers.py`)
Capas $m = 1,\dots,M$. $V^0 = X$ (shape $n\times n_0$). $W^m$ tiene shape $(n_{m-1}, n_m)$ y $b^m$ shape $(1, n_m)$:
$$Z^m = V^{m-1}W^m + b^m,\qquad V^m = \theta_m(Z^m),\qquad O = V^M$$
La **columna** $j$ de $W^m$ son los pesos que entran a la neurona $j$. Relación con la cátedra: $W^{\text{cátedra}}_{ij}$ (de la neurona $j$ a la $i$) $=$ `W[j, i]` del código.

### §3.2 Retropropagación por dato (notación de la cátedra)
- Salida: $\delta_i = (\zeta_i - O_i)\,\theta'(h_i)$, $\;\Delta W_{ij} = \eta\,\delta_i\,V_j^{M-1}$.
- Oculta: $\delta_j = \theta'(h_j)\sum_i W_{ij}\,\delta_i$, $\;\Delta w_{jk} = \eta\,\delta_j\,V_k^{m-1}$.

### §3.3 Retropropagación matricial (lo que está implementado en `Dense.backward` / `Network.backward`)
Sea $\Delta^m \equiv \partial L/\partial Z^m$ (shape $n\times n_m$).
$$\Delta^M = \frac{\partial L}{\partial O}\odot \theta_M'(Z^M)\qquad(\text{MSE del código: } \tfrac{2}{nm}(O - \zeta)\odot\theta_M'(Z^M);\ \text{softmax+CCE: } \tfrac1n(O - y))$$
$$\Delta^m = \big(\Delta^{m+1}\,(W^{m+1})^\top\big)\odot\theta_m'(Z^m),\quad m = M-1,\dots,1$$
$$\frac{\partial L}{\partial W^m} = (V^{m-1})^\top\Delta^m,\qquad \frac{\partial L}{\partial b^m} = \textstyle\sum_{\text{filas}}\Delta^m$$
Relación con §3.2 (una muestra, una salida, MSE del código): $\delta = -\tfrac12\Delta$, así que el paso del código con $\eta_c$ equivale al de la cátedra con $\eta = 2\eta_c$ (ver §2.4).

### §3.4 Verificación de gradiente (ya implementada en `tests/test_gradients.py`)
$$g^{num} = \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon},\ \varepsilon = 10^{-5};\qquad \frac{|g - g^{num}|}{\max(|g|, |g^{num}|, 10^{-8})} < 10^{-6}$$
El test además verifica que el chequeo **detecte** 4 bugs típicos introducidos a propósito. Toda extensión nueva que toque gradientes (L2 en F04) agrega su caso a este test.

### §3.5 Inicialización
Pesos **aleatorios, nunca todos iguales** (problema de simetría). En el código: `uniform` (default $U(-0.5, 0.5)$), `xavier` (default de `Network`), `he`; bias en 0.

> ⚠️ La escala importa. Con entradas ±1 (XOR), pesos $U(-0.1, 0.1)$ dejan la red en la meseta cercana al origen (salida ≈ 0) y no aprende en miles de épocas; con escala ~1 o Xavier converge en decenas (verificado numéricamente). Con Xavier, `[2,2,1]` resuelve XOR en 17/20 semillas y `[2,3,2,1]` en 20/20 (`tests/test_validation.py`): las 3 fallas son el mínimo local clásico de XOR. Buen material para la slide de validación.

---

## §4 Optimizadores — Clase 12.1 y Optimización No Lineal

Notación: $g_t = \partial L/\partial\theta$ en el paso $t$ (elemento a elemento; $g_t^2$ es cuadrado elemento a elemento).

| Optimizador | Actualización | Defaults |
|---|---|---|
| GD / SGD | $\theta \leftarrow \theta - \eta\,g_t$ | — |
| Momentum (cátedra) | $\Delta\theta(t{+}1) = -\eta\,g_t + \alpha\,\Delta\theta(t)$; $\theta \leftarrow \theta + \Delta\theta(t{+}1)$ | $\alpha = 0.9$ (0.8–0.9) |
| AdaGrad (opc.) | $G_t = G_{t-1} + g_t^2$; $\theta \leftarrow \theta - \eta\,g_t/(\sqrt{G_t} + \epsilon)$ | $\epsilon = 10^{-8}$ |
| RMSProp (cátedra) | $S_t = \gamma S_{t-1} + (1-\gamma)g_t^2$; $\theta \leftarrow \theta - \eta\,g_t/\sqrt{S_t + \epsilon}$ | $\gamma = 0.9$, $\epsilon = 10^{-8}$ |
| Adam (Kingma & Ba) | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$; $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$; $\hat m_t = m_t/(1-\beta_1^t)$; $\hat v_t = v_t/(1-\beta_2^t)$; $\theta \leftarrow \theta - \eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ | $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ |

El estado ($\Delta\theta$, $G$, $S$, $m$, $v$, $t$) es **por tensor de parámetros**: como `step` recibe listas paralelas `params`/`grads`, el estado se indexa por posición en la lista (se crea en el primer `step`). Todo se actualiza **in place** (`p -= ...`, nunca `p = p - ...`). El estado se guarda en el checkpoint (F07).

### §4.1 η adaptativo (cátedra) — callback por época
Sobre el error de entrenamiento de cada época $E_e$:
$$\Delta\eta = \begin{cases} +a & \text{si } E \text{ decreció } k \text{ épocas seguidas}\\ -b\,\eta & \text{si } E \text{ creció } k' \text{ épocas seguidas}\\ 0 & \text{en otro caso}\end{cases}$$
Después de cada cambio se reinician los contadores. Parámetros: $a$, $b\in(0,1)$, $k$, $k'$. Combinable con cualquier optimizador (modifica `optimizer.lr`). Variante permitida (footnote de la clase): "consistente" = variación menor a P % en K épocas.

---

## §5 Regularización — Clase 13

- **L2 / weight decay:** $E_{reg} = E + \frac{\lambda}{2}\sum_m\lVert W^m\rVert_F^2 \Rightarrow g_{W^m} \mathrel{+}= \lambda W^m$. No se penalizan bias.
- **Early stopping:** monitorear `val_loss` (o una métrica de validación); cortar si no mejora más de `min_delta` durante `patience` épocas; restaurar los mejores pesos (ya implementado en `EarlyStopping`).
- **Data augmentation** (solo train, nunca val/test): ruido gaussiano $x' = \text{clip}(x + N(0,\sigma^2))$ al rango de la feature, nuevo en cada época; traslaciones de ±k píxeles si los dígitos son imágenes. **Cuidado con no cambiar la etiqueta** (p. ej. no rotar un 6 hasta parecer 9).

---

## §6 Métricas — Clase "Métricas y Sobreajuste"

Matriz de confusión: **filas = clase real, columnas = clase predicha** (convención de la cátedra).

| | Pred. positivo | Pred. negativo |
|---|---|---|
| **Real positivo** | TP | FN |
| **Real negativo** | FP | TN |

- Accuracy $= \frac{TP+TN}{TP+TN+FP+FN}$ · Precision $= \frac{TP}{TP+FP}$ · Recall = TPR $= \frac{TP}{TP+FN}$ · F1 $= \frac{2PR}{P+R}$ · FPR $= \frac{FP}{FP+TN}$
- $F_\beta = \frac{(1+\beta^2)PR}{\beta^2P + R}$ (β > 1 prioriza recall).
- Multiclase: métricas por clase (uno contra todos) + **macro** (promedio simple entre clases). Accuracy multiclase = traza / total.
- División por cero → 0.0 y se marca en el resultado (`"undefined": true`).
- Regresión (distillation sobre probabilidades): MSE, RMSE, MAE.

### §6.1 Diagnóstico (resumen de la clase)

| Error en train | Error en test | Diagnóstico |
|---|---|---|
| Alto | Alto | Underfitting |
| Bajo | Alto | Overfitting |
| Bajo | Bajo | Buen modelo |

---

## §7 Umbral de decisión (Ej1)

Dado un score $s\in[0,1]$ y umbral $t$: predicción positiva si $s \ge t$.
- Barrido $t \in \{0.00, 0.01, \dots, 1.00\}$ → TPR, FPR, precision, recall, F1, $F_2$, costo.
- **ROC**: TPR vs FPR; **AUC** por trapecios ordenando por FPR.
- **PR**: precision vs recall; **AP** $= \sum_k (R_k - R_{k-1})P_k$.
- Criterios de selección (se reportan todos, se recomienda uno y se justifica):
  1. $\arg\max_t F_1$ · 2. $\arg\max_t F_2$ · 3. Youden $J = TPR - FPR$ · 4. Costo esperado mínimo $C(t) = c_{FN}\,FN(t) + c_{FP}\,FP(t)$ con costos parametrizables · 5. Máxima precision sujeta a recall ≥ $r_{min}$.
- El umbral se elige en **validación** (idealmente promediando folds) y se reporta **una vez** en test.

---

## §8 Normalización — Clase "Métricas y Sobreajuste"

- Min-max a $[a,b]$: $x' = a + \frac{(x - \min)(b-a)}{\max - \min}$ (columna constante → $a$ y se avisa).
- Z-score: $x' = (x - \bar x)/s_x$ ($s_x = 0$ → 0).
- Unit length: $x' = x/\lVert x\rVert_2$ (por fila / muestra).
- Estadísticos **siempre** ajustados con el train de la partición (ver Constitución C5).
- Si la salida esperada no cae en la imagen de $\theta$ de salida, se escala $\zeta$ a esa imagen y se invierte al predecir.

---

## §9 Verificación a mano (V-05)

La iteración oficial es la de `docs/verificacion_manual.md` (red `[2,2,1]` tanh, MSE del código, GD η = 0.1), reproducida exactamente por `tests/test_validation.py::test_verificacion_manual_221`. **Pendiente humano:** rehacerla en papel con calculadora (F08-T3) y, si el equipo quiere, un segundo fixture `[2,3,2,1]` con pesos propios.