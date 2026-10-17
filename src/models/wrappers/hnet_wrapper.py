"""
Envoltorio de H-Net para el motor de inferencia.

La importación de ``hnet_compat`` debe preceder a cualquier importación de
``hnet``: aplica los shims de compatibilidad y registra la ruta del paquete.
"""

from src.models.architectures import hnet_compat  # noqa: F401
