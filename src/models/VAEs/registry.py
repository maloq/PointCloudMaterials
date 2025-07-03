ENCODERS: dict[str, type] = {}
DECODERS: dict[str, type] = {}

def register_encoder(name: str):
    def wrap(cls):
        ENCODERS[name] = cls
        return cls
    return wrap

def register_decoder(name: str):
    def wrap(cls):
        DECODERS[name] = cls
        return cls
    return wrap