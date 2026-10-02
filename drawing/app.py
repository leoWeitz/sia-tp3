"""Aplicación interactiva para probar el modelo de dígitos dibujando a mano alzada.

Zero-dependencies: utiliza únicamente la librería estándar de Python (`http.server`,
`urllib`, `json`) junto con `numpy` y los modelos entrenados en `models/`.

Características:
- Lienzo 28x28 interactivo con zoom (336x336 px) y vista previa 1:1 real (28x28 px).
- Selector de modelo en vivo: Ej3 Best (ReLU, 98.36%) vs Ej2 Best (Tanh, sin ochos).
- Inferencia en tiempo real con barras de probabilidad para los 10 dígitos.
- Auto-centrado por centro de masa opcional (estándar MNIST).

Uso:
    python -m drawing.app [--port 8080] [--no-browser]
"""

import argparse
import json
import socket
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from core.serialization import load_checkpoint

# Modelos disponibles
MODELS_DIR = Path("models")
HTML_FILE = Path(__file__).parent / "index.html"

AVAILABLE_MODELS = {
    "ej3_best": {
        "name": "Ej3 Best (ReLU [784, 256, 128, 10] · Acc Test 98.36%)",
        "path": MODELS_DIR / "ej3_best" / "model.npz",
        "description": "Modelo final con dataset completo, ReLU y RandomShift.",
    },
    "ej2_best": {
        "name": "Ej2 Best (Tanh [784, 128, 10] · Acc Test 88.31%)",
        "path": MODELS_DIR / "ej2_best" / "model.npz",
        "description": "Entrenado sin el dígito 8 (falla sistemáticamente los 8s).",
    },
}

DEFAULT_MODEL_KEY = "ej3_best"
LOADED_NETWORKS: dict[str, Any] = {}


def get_network(model_key: str):
    """Carga y cachea la red neuronal solicitada."""
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Modelo desconocido: {model_key}")
    if model_key not in LOADED_NETWORKS:
        path = AVAILABLE_MODELS[model_key]["path"]
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {path}")
        net, _, _ = load_checkpoint(path)
        LOADED_NETWORKS[model_key] = net
    return LOADED_NETWORKS[model_key]


def center_digit_mass(img_28x28: np.ndarray) -> np.ndarray:
    """Centra la imagen en función de su centro de masa (estándar MNIST)."""
    rows = np.any(img_28x28 > 0.05, axis=1)
    cols = np.any(img_28x28 > 0.05, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img_28x28

    total_mass = np.sum(img_28x28)
    if total_mass == 0:
        return img_28x28

    y_coords, x_coords = np.indices((28, 28))
    cy = np.sum(y_coords * img_28x28) / total_mass
    cx = np.sum(x_coords * img_28x28) / total_mass

    shift_y = int(np.round(14.0 - cy))
    shift_x = int(np.round(14.0 - cx))

    out = np.zeros_like(img_28x28)
    src_y1 = max(0, -shift_y)
    src_y2 = min(28, 28 - shift_y)
    dst_y1 = max(0, shift_y)
    dst_y2 = min(28, 28 + shift_y)

    src_x1 = max(0, -shift_x)
    src_x2 = min(28, 28 - shift_x)
    dst_x1 = max(0, shift_x)
    dst_x2 = min(28, 28 + shift_x)

    out[dst_y1:dst_y2, dst_x1:dst_x2] = img_28x28[src_y1:src_y2, src_x1:src_x2]
    return out


class DrawingHandler(BaseHTTPRequestHandler):
    """Manejador HTTP para la app de dibujo."""

    def log_message(self, format: str, *args: Any) -> None:
        # Silenciar logs para no inundar la terminal durante el dibujo
        pass

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            if not HTML_FILE.exists():
                self.send_error(500, "Archivo index.html no encontrado")
                return
            content = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/api/models":
            models_info = {
                k: {"name": v["name"], "description": v["description"]}
                for k, v in AVAILABLE_MODELS.items()
            }
            body = json.dumps(models_info).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "Ruta no encontrada")

    def do_POST(self) -> None:
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                model_key = data.get("model", DEFAULT_MODEL_KEY)
                raw_pixels = np.array(data.get("pixels", []), dtype=np.float64).reshape((28, 28))
                autocenter = data.get("autocenter", True)

                # Centrado de masa si está activo
                if autocenter:
                    processed = center_digit_mass(raw_pixels)
                else:
                    processed = raw_pixels

                X = processed.reshape(1, 784)
                net = get_network(model_key)
                scores = net.predict(X)[0]

                prediction = int(np.argmax(scores))
                confidence = float(scores[prediction])

                resp = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "scores": [float(s) for s in scores],
                    "centered_pixels": [float(p) for p in processed.ravel()],
                }

                resp_bytes = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
            except Exception as e:
                err_bytes = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_error(404, "Endpoint no encontrado")


def find_free_port(start_port: int = 8080) -> int:
    """Busca un puerto libre a partir de start_port."""
    port = start_port
    while port < start_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
            port += 1
    return start_port


def main() -> int:
    parser = argparse.ArgumentParser(description="App web interactiva de dibujo de dígitos")
    parser.add_argument(
        "--port", type=int, default=8080, help="Puerto del servidor (default: 8080)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="No abrir automáticamente el navegador"
    )
    args = parser.parse_args()

    port = find_free_port(args.port)
    url = f"http://localhost:{port}"

    print(f"Cargando modelo por defecto ({DEFAULT_MODEL_KEY})...")
    get_network(DEFAULT_MODEL_KEY)
    print("✓ Modelo cargado exitosamente.")

    server = ThreadingHTTPServer(("127.0.0.1", port), DrawingHandler)
    print("\n=======================================================")
    print(f"  🎨 Servidor de Dibujo Iniciado en: {url}")
    print("  Presioná Ctrl+C para detener el servidor")
    print("=======================================================\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDeteniendo servidor...")
        server.shutdown()
        return 0


if __name__ == "__main__":
    sys.exit(main())
