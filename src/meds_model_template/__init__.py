"""``meds-model-template``: a Copier template for MEDS models with a mandated 5-step CLI.

This package is a *thin bootstrap*. The real payload is the Copier template under ``template/`` in the
source repository; this package only exposes a convenience command (``meds-model-new``) that shells out
to Copier so users who ``pip install meds-model-template`` can scaffold a new model without knowing the
Copier invocation.
"""

__version__ = "0.1.0"
