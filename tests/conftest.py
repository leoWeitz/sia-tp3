"""Configuración común de pytest."""

import matplotlib

# Backend sin ventanas, fijado antes de que algún test importe pyplot: con TkAgg, crear
# una figura falla de forma intermitente en Windows (TclError: Can't find a usable init.tcl).
matplotlib.use("Agg")
