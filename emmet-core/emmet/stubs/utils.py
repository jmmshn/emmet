""" Utilities to enable stubbing in JSON schema into pydantic for Monty """
from typing import Dict, Any, TypeVar, Union, Optional, Tuple, List, Set, Sequence
from typing import get_type_hints
from numpy import ndarray
from monty.json import MSONable, MontyDecoder
from pydantic import create_model, Field

built_in_primitives = (bool, int, float, complex, range, str, bytes, None)
prim_to_type_hint = {list: List, tuple: Tuple, dict: Dict, set: Set}

STUBS = {}  # Central location for Pydantic Stub classes


def patch_msonable(monty_cls):
    """
    Patch's an MSONable class so it can be used in pydantic models

    monty_cls: A MSONable class
    """

    if not issubclass(monty_cls, MSONable):
        raise ValueError("Must provide an MSONable class to wrap")

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_monty

    @classmethod
    def validate_monty(cls, v):
        """
        Stub validator for MSONable
        """
        if isinstance(v, cls):
            return v
        elif isinstance(v, dict):
            # Relegate to Monty
            new_obj = MontyDecoder().process_decoded(v)
            if not isinstance(new_obj, cls):
                raise ValueError(f"Wrong dict for {cls.__name__}")
            return new_obj
        else:
            raise ValueError(f"Must provide {cls.__name__} or Dict version")

    setattr(monty_cls, "validate_monty", validate_monty)
    setattr(monty_cls, "__get_validators__", __get_validators__)
    setattr(monty_cls, "__pydantic_model__", STUBS[monty_cls])


def use_model(monty_cls, pydantic_model, add_monty=True):
    """
    Use a provided pydantic model to describe a Monty MSONable class
    """

    if add_monty:
        monty_props = {
            "@class": Field(
                monty_cls.__name__,
                description="The formal class name for serialization lookup",
            ),
            "@module": Field(
                monty_cls.__module__, description="The module this class is defined in"
            ),
        }
        pydantic_model.__fields__.update(monty_props)
    STUBS[monty_cls] = pydantic_model
    patch_msonable(monty_cls)


def __make_pydantic(cls):
    """
    Temporary wrapper function to convert an MSONable class into a PyDantic
    Model for the sake of building schemas
    """

    if any(cls == T for T in built_in_primitives):
        return cls

    if cls in prim_to_type_hint:
        return prim_to_type_hint[cls]

    if cls == Any:
        return Any

    if type(cls) == TypeVar:
        return cls

    if hasattr(cls, "__origin__") and hasattr(cls, "__args__"):

        args = tuple(__make_pydantic(arg) for arg in cls.__args__)
        if cls.__origin__ == Union:
            return Union.__getitem__(args)

        if cls.__origin__ == Optional and len(args) == 1:
            return Optional.__getitem__(args)

        if cls._name == "List":
            return List.__getitem__(args)

        if cls._name == "Tuple":
            return Tuple.__getitem__(args)

        if cls._name == "Set":
            return Set.__getitem__(args)

        if cls._name == "Sequence":
            return Sequence.__getitem__(args)

    if issubclass(cls, MSONable):
        if cls.__name__ not in STUBS:

            monty_props = {
                "@class": Field(
                    cls.__name__,
                    description="The formal class name for serialization lookup",
                ),
                "@module": Field(
                    cls.__module__, description="The module this class is defined in"
                ),
            }
            props = {
                field_name: (__make_pydantic(field_type), ...)
                for field_name, field_type in get_type_hints(cls.__init__).items()
            }

            STUBS[cls] = create_model(cls.__name__, **monty_props, **props)
        return STUBS[cls]

    if cls == ndarray:
        return List[Any]

    return cls
