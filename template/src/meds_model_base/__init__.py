"""``meds_model_base`` — the vendored, template-managed contract for a MEDS model repository.

This package owns the mandated CLI contract and its default implementations. It is copied verbatim into
every repository generated from ``MEDS_model_template`` and is refreshed by ``copier update``; downstream
users should not edit it (edit ``src/<your_model>/`` instead).

Public surface:

- :mod:`meds_model_base.schemas` — canonical + template schemas and validators.
- :mod:`meds_model_base.steps` — ``StepName``, the ``MEDSModelStep`` ABCs, and the default step classes.
- :mod:`meds_model_base.dispatch` — ``make_cli`` (the ``meds-model`` dispatcher) and ``register_resolvers``.

Note: importing heavy submodules (``steps``, ``lightning``, ``dispatch``) pulls in torch / lightning /
meds-torch-data; :mod:`meds_model_base.schemas` is dependency-light and safe to import on its own.
"""

__version__ = "0.1.0"
