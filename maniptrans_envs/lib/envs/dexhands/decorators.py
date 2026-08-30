from .factory import DexHandFactory


def register_dexhand(dexhand_type):
    def decorator(cls):
        DexHandFactory.register(dexhand_type, cls)
        return cls

    return decorator
