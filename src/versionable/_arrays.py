"""Array dtype canonicalisation and runtime validation.

``NDArray[np.float64]``-style annotations carry a concrete numpy dtype.  That
dtype is hash-significant (ADR-0002): the canonical type name is
``ndarray[float64]``, shape is erased, and the declared dtype is enforced at
save and load time so the hash fingerprints something the runtime actually
guarantees.

Dtype tokens are a **closed set** — the table in ``conformance/GRAMMAR.md``.
Tokens come from the explicit mapping below rather than ``numpy.dtype.name`` so
that a numpy release cannot silently change a schema hash, and so that a dtype
with no cross-language equivalent is rejected instead of hashing to a token no
other implementation knows.

Enforcement is deliberately asymmetric: casts that numpy considers *safe*
(``np.can_cast(..., casting="safe")``) are applied silently, everything else
raises :class:`~versionable.errors.DtypeMismatchError`.
"""

from __future__ import annotations

import types
import typing
from typing import Any

from versionable._numpy_compat import _np as np
from versionable.errors import DtypeContext, DtypeMismatchError, UnsupportedTypeError

# numpy is an optional dependency, so it cannot be imported for annotations —
# ``numpy.dtype`` values are therefore typed ``Any``.  This module is imported by
# ``_hash``, which must stay importable (and type-checkable) without numpy.
NumpyDtype = Any


def _buildDtypeTokens() -> dict[Any, str]:
    """Build the closed dtype→token table (empty when numpy is absent)."""
    if np is None:
        return {}
    pairs: tuple[tuple[Any, str], ...] = (
        (np.bool_, "bool"),
        (np.int8, "int8"),
        (np.int16, "int16"),
        (np.int32, "int32"),
        (np.int64, "int64"),
        (np.uint8, "uint8"),
        (np.uint16, "uint16"),
        (np.uint32, "uint32"),
        (np.uint64, "uint64"),
        (np.float16, "float16"),
        (np.float32, "float32"),
        (np.float64, "float64"),
        (np.complex64, "complex64"),
        (np.complex128, "complex128"),
    )
    return {np.dtype(scalar): token for scalar, token in pairs}


_DTYPE_TOKENS = _buildDtypeTokens()


def isNdarrayOrigin(origin: Any) -> bool:
    """Return True if *origin* is ``numpy.ndarray``.

    Uses module/qualname rather than an ``is`` comparison so the check works
    without importing numpy (the hash module must stay importable when numpy
    is not installed).
    """
    return getattr(origin, "__module__", "") == "numpy" and getattr(origin, "__qualname__", "") == "ndarray"


def canonicalArrayName(args: tuple[Any, ...]) -> str:
    """Return the canonical name for an ``ndarray[...]`` alias with *args*.

    Raises:
        UnsupportedTypeError: The annotation declares a dtype outside the closed
            token table.
    """
    dtype = dtypeFromArgs(args)
    return "ndarray" if dtype is None else f"ndarray[{dtypeToken(dtype)}]"


def dtypeToken(dtype: NumpyDtype) -> str:
    """Return the canonical grammar token for *dtype*.

    Raises:
        UnsupportedTypeError: *dtype* is outside the closed token table.
    """
    token = _DTYPE_TOKENS.get(dtype)
    if token is None:
        supported = ", ".join(sorted(_DTYPE_TOKENS.values()))
        raise UnsupportedTypeError(
            f"numpy dtype '{dtype}' has no canonical grammar token, so it cannot appear in a schema hash. "
            f"Supported array dtypes: {supported}. "
            f"Annotate the field as a bare 'np.ndarray' to store this data without declaring a dtype."
        )
    return token


def dtypeFromArgs(args: tuple[Any, ...]) -> NumpyDtype | None:
    """Return the dtype declared by the arguments of an ``ndarray[...]`` alias.

    ``npt.NDArray[np.float64]`` expands to
    ``np.ndarray[tuple[Any, ...], np.dtype[np.float64]]``; the shape argument is
    ignored.  Returns ``None`` when no concrete dtype is declared — an
    unparametrised ``npt.NDArray`` (whose scalar argument is still a TypeVar),
    ``NDArray[Any]``, or an abstract scalar type such as ``np.floating``.
    """
    if np is None:
        return None
    for arg in args:
        scalarType = _scalarType(arg)
        if scalarType is None:
            continue
        try:
            dtype: NumpyDtype = np.dtype(scalarType)
        except TypeError:
            # Abstract scalar types (np.floating, np.generic) have no dtype.
            return None
        return dtype
    return None


def declaredDtype(fieldType: Any) -> NumpyDtype | None:
    """Return the dtype declared by an array annotation, or ``None``.

    Unwraps ``Annotated`` and unions, so ``NDArray[np.float64] | None`` declares
    ``float64``.  A union declaring two different dtypes declares neither.
    """
    if np is None:
        return None

    origin = typing.get_origin(fieldType)
    args = typing.get_args(fieldType)

    if origin is typing.Annotated:
        return declaredDtype(args[0])

    if origin is typing.Union or origin is types.UnionType:
        found = {declaredDtype(arg) for arg in args if arg is not type(None)}
        found.discard(None)
        return next(iter(found)) if len(found) == 1 else None

    if isNdarrayOrigin(origin):
        return dtypeFromArgs(args)

    return None


def coerceArrayDtype(
    value: Any,
    fieldType: Any,
    *,
    fieldPath: str = "",
    context: DtypeContext = "save",
) -> Any:
    """Validate *value* against the dtype declared by *fieldType*.

    Args:
        value: Candidate field value.  Non-arrays are returned untouched.
        fieldType: The declared type annotation for the field.
        fieldPath: Dotted path of the field, used in the error message.
        context: Whether this runs on ``save`` or ``load``.

    Returns:
        *value* unchanged, or a safely cast copy when its dtype differs from the
        declared one but the cast loses nothing.

    Raises:
        DtypeMismatchError: The array's dtype cannot be safely cast to the
            declared dtype.
    """
    if np is None or not isinstance(value, np.ndarray):
        return value
    return coerceToDtype(value, declaredDtype(fieldType), fieldPath=fieldPath, context=context)


def coerceToDtype(
    value: Any,
    dtype: NumpyDtype | None,
    *,
    fieldPath: str = "",
    context: DtypeContext = "save",
) -> Any:
    """Validate *value* against an already-resolved declared *dtype*.

    Used where the declared dtype is known but the annotation is not to hand —
    lazy HDF5 sentinels and session-backed datasets.  A ``None`` dtype (a bare,
    undeclared ``ndarray`` field) accepts anything.
    """
    if np is None or not isinstance(value, np.ndarray):
        return value
    target = resolveCast(value.dtype, dtype, fieldPath=fieldPath, context=context)
    if target is None:
        return value
    cast: Any = value.astype(target)
    return cast


def resolveCast(
    actual: NumpyDtype,
    declared: NumpyDtype | None,
    *,
    fieldPath: str = "",
    context: DtypeContext = "save",
) -> NumpyDtype | None:
    """Return the dtype an array of *actual* dtype must be cast to, or ``None``.

    Lets a caller that holds a dtype but not yet the data — an HDF5 dataset
    header — reject an unsafe mismatch immediately and defer the safe cast to
    the moment the array is materialized.

    Raises:
        DtypeMismatchError: Casting *actual* to *declared* would lose data.
    """
    if np is None or declared is None or actual == declared:
        return None

    # Zero-width flexible dtypes ('<U0') are not declarable — the token table
    # rejects them — but never cast to one anyway: astype() would truncate every
    # item to zero length.
    if declared.itemsize != 0 and np.can_cast(actual, declared, casting="safe"):
        return declared

    raise DtypeMismatchError(
        declared=_tokenOrName(declared),
        actual=_tokenOrName(actual),
        fieldPath=fieldPath,
        context=context,
    )


def _tokenOrName(dtype: NumpyDtype) -> str:
    """Return the canonical token for *dtype*, falling back to its numpy name.

    Only used for error messages, where the dtype found in memory or on disk
    still has to be nameable even when it is outside the token table.
    """
    token: str | None = _DTYPE_TOKENS.get(dtype)
    return token if token is not None else str(dtype.name)


def _scalarType(arg: Any) -> type | None:
    """Return the numpy scalar type carried by an ``ndarray`` type argument.

    Accepts both the standard ``np.dtype[np.float64]`` form and a bare scalar
    type (``np.ndarray[Any, np.float64]``).  Returns ``None`` for the shape
    argument, ``Any``, and TypeVars.
    """
    if np is None:
        return None
    if typing.get_origin(arg) is np.dtype:
        inner = typing.get_args(arg)
        return _concreteType(inner[0]) if inner else None
    scalar = _concreteType(arg)
    if scalar is not None and issubclass(scalar, np.generic):
        return scalar
    return None


def _concreteType(arg: Any) -> type | None:
    """Return *arg* if it is a concrete type, else ``None`` (``Any``, TypeVars, aliases)."""
    if arg is Any or not isinstance(arg, type):
        return None
    return arg
