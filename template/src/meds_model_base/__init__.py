"""``meds_model_base`` — the vendored, template-managed contract for a MEDS model repository.

This package owns the mandated CLI contract and its default implementations. It is copied verbatim into
every repository generated from ``MEDS_model_template`` and is refreshed by ``copier update``; downstream
users should not edit it (edit ``src/<your_model>/`` instead).

Public surface:

- :mod:`meds_model_base.commands` — ``CommandName``, the ``MEDSModelCommand`` ABCs, source arbitration,
  and the default implementations of the five commands.
- :mod:`meds_model_base.manifest` — artifact manifests: atomic publication, provenance, input validation.
- :mod:`meds_model_base.schemas` — canonical + template schemas and validators.
- :mod:`meds_model_base.tasks` — turning an external task file into split label parquets.
- :mod:`meds_model_base.dispatch` — ``make_cli`` (the ``meds-model`` dispatcher) and ``register_resolvers``.
- :mod:`meds_model_base.meds_dev` — ``meds-model-add-to-meds-dev``: installs ``model.yaml`` into a
  MEDS-DEV checkout. Packaging tooling, not part of the command contract.

Note: importing the default command implementations (or ``lightning`` / ``dispatch``) pulls in torch /
lightning / meds-torch-data. :mod:`meds_model_base.schemas`, :mod:`meds_model_base.manifest` and
:mod:`meds_model_base.commands.base` are dependency-light and safe to import on their own.
"""

__version__ = "0.1.0"
