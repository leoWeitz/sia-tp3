# TP3 SIA — Perceptrón Simple y Multicapa

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest                              # Hay un test inicial que simplemente chequea que se puedan importar los módulos 
                                    # y que los paquetes necesarios estén instalados
```
## CI/CD

Por cada PR o push se corre un script que clona el repo, instala las dependencias de "requirements.txt" y corre un pytest.


Convenciones y reglas del proyecto: ver `CLAUDE.md`.