from __future__ import annotations


ENCODERS: dict[str, type] = {}


def register_encoder(name: str):
    def wrap(cls):
        existing = ENCODERS.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Encoder name {name!r} is already registered by {existing.__name__}; "
                f"cannot also register {cls.__name__}."
            )
        ENCODERS[name] = cls
        return cls

    return wrap

