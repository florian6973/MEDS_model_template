"""Stable command vocabulary and the six supported MEDS model profiles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any


class CommandName(StrEnum):
    preprocess_data = "preprocess_data"
    pretrain = "pretrain"
    infer = "infer"
    supervised_train = "supervised_train"
    predict = "predict"


class ArtifactType(StrEnum):
    patients = "patients"
    pretrained_model = "pretrained_model"
    inference = "inference"
    supervised_model = "supervised_model"
    predictions = "predictions"


MODEL_PROFILES: dict[str, tuple[CommandName, ...]] = {
    "supervised": (CommandName.preprocess_data, CommandName.supervised_train, CommandName.predict),
    "finetune": (
        CommandName.preprocess_data,
        CommandName.pretrain,
        CommandName.supervised_train,
        CommandName.predict,
    ),
    "probe": tuple(CommandName),
    "zero_shot_direct": (CommandName.preprocess_data, CommandName.pretrain, CommandName.predict),
    "zero_shot_materialized": (
        CommandName.preprocess_data,
        CommandName.pretrain,
        CommandName.infer,
        CommandName.predict,
    ),
    "packaged": (CommandName.preprocess_data, CommandName.predict),
}


class Command(ABC):
    """A model-owned implementation of one contract command."""

    name: CommandName

    @abstractmethod
    def run(self, arguments: Mapping[str, Any]) -> Path:
        """Run the command and return its published artifact directory."""

    def __call__(self, arguments: Mapping[str, Any]) -> Path:
        return self.run(arguments)


class ModelRepository:
    """Validate and dispatch the model-owned implementations for a selected model profile."""

    def __init__(self, profile: str, commands: Mapping[CommandName, type[Command]]) -> None:
        if profile not in MODEL_PROFILES:
            expected_profiles = ", ".join(MODEL_PROFILES)
            raise ValueError(f"Unknown model profile {profile!r}; expected one of {expected_profiles}")
        expected = set(MODEL_PROFILES[profile])
        actual = set(commands)
        if actual != expected:
            raise ValueError(
                f"The {profile!r} model profile requires {sorted(expected)}, but commands.py declares "
                f"{sorted(actual)}"
            )
        self.profile = profile
        self.commands = dict(commands)

    @property
    def profile_commands(self) -> Sequence[CommandName]:
        return MODEL_PROFILES[self.profile]

    def run(self, name: CommandName | str, arguments: Mapping[str, Any]) -> Path:
        command_name = CommandName(name)
        if command_name not in self.commands:
            raise ValueError(f"{command_name.value} is not part of the {self.profile} model profile")
        return self.commands[command_name]()(arguments)
