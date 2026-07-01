"""The four reference model profiles, built on the shared modern stack (Hydra + Lightning + MTD).

Each profile provides a ``LightningModule`` (and, where the step orchestration genuinely differs, a step
subclass). A generated repo's ``src/<model_slug>/model.py`` subclasses the chosen profile's module so the
user has a small, owned surface to edit while the contract stays in ``meds_model_base``.

- :mod:`~meds_model_base.profiles.supervised` — ``SupervisedClassifier`` (supervised-basic ``{a,c,e}``).
- :mod:`~meds_model_base.profiles.autoregressive` — ``AutoregressiveModel`` (zero-shot ``{a,b,d,e}``).
- :mod:`~meds_model_base.profiles.every_query` — ``EveryQueryModel`` (query pretraining ``{a,b,e}``).
- :mod:`~meds_model_base.profiles.motor` — ``MotorModel`` (time-to-event fine-tune ``{a,b,c,e}``).
"""
