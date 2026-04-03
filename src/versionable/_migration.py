"""Declarative and imperative migration system.

Migrations transform serialized data from older versions to the current
schema.  They are declared as a ``Migrate`` inner class on Versionable
subclasses::

    @dataclass
    class Config(Versionable, version=3, hash='abc123'):
        name: str
        timeout: int

        class Migrate:
            v2 = Migration().rename('old_name', 'name')
            v1 = Migration().add('timeout', default=30).then(v2)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from versionable.errors import MigrationError, UpgradeRequiredError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Migration operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RenameOp:
    old: str
    new: str


@dataclass(frozen=True)
class _AddOp:
    field: str
    default: Any


@dataclass(frozen=True)
class _DropOp:
    field: str


@dataclass(frozen=True)
class _ConvertOp:
    field: str
    via: Callable[[Any], Any]
    reverse: Callable[[Any], Any] | None = None


@dataclass(frozen=True)
class _DeriveOp:
    field: str
    fromField: str
    via: Callable[[Any], Any]


@dataclass(frozen=True)
class _SplitOp:
    field: str
    into: dict[str, Callable[[Any], Any]]


@dataclass(frozen=True)
class _MergeOp:
    fields: list[str]
    into: str
    via: Callable[..., Any]


@dataclass(frozen=True)
class _RequiresUpgradeOp:
    pass


type MigrationOp = _RenameOp | _AddOp | _DropOp | _ConvertOp | _DeriveOp | _SplitOp | _MergeOp | _RequiresUpgradeOp


# ---------------------------------------------------------------------------
# Migration class (builder pattern)
# ---------------------------------------------------------------------------


class Migration:
    """Declarative migration builder.

    Chain operations to describe how to transform data from an older
    version to a newer one::

        v2 = (
            Migration()
            .rename('old_name', 'new_name')
            .add('new_field', default=0)
            .drop('removed_field')
        )
    """

    def __init__(self) -> None:
        self._ops: list[MigrationOp] = []

    def rename(self, old: str, new: str) -> Migration:
        """Rename a field."""
        self._ops.append(_RenameOp(old, new))
        return self

    def add(self, field: str, *, default: Any) -> Migration:
        """Add a new field with a default value."""
        self._ops.append(_AddOp(field, default))
        return self

    def drop(self, field: str) -> Migration:
        """Remove a field."""
        self._ops.append(_DropOp(field))
        return self

    def convert(
        self,
        field: str,
        *,
        via: Callable[[Any], Any],
        reverse: Callable[[Any], Any] | None = None,
    ) -> Migration:
        """Convert a field's value using a transformation function."""
        self._ops.append(_ConvertOp(field, via, reverse))
        return self

    def derive(
        self,
        field: str,
        *,
        from_: str,
        via: Callable[[Any], Any],
    ) -> Migration:
        """Derive a new field from an existing one."""
        self._ops.append(_DeriveOp(field, from_, via))
        return self

    def split(
        self,
        field: str,
        *,
        into: dict[str, Callable[[Any], Any]],
    ) -> Migration:
        """Split one field into multiple fields."""
        self._ops.append(_SplitOp(field, into))
        return self

    def merge(
        self,
        fields: list[str],
        *,
        into: str,
        via: Callable[..., Any],
    ) -> Migration:
        """Merge multiple fields into one."""
        self._ops.append(_MergeOp(fields, into, via))
        return self

    def requiresUpgrade(self) -> Migration:
        """Mark this migration as requiring in-place file modification."""
        self._ops.append(_RequiresUpgradeOp())
        return self

    def then(self, other: Migration) -> Migration:
        """Chain another migration after this one."""
        combined = Migration()
        combined._ops = list(self._ops) + list(other._ops)
        return combined

    @property
    def ops(self) -> list[MigrationOp]:
        """Return the list of migration operations (read-only access)."""
        return list(self._ops)


# ---------------------------------------------------------------------------
# Imperative migration decorator
# ---------------------------------------------------------------------------


class _ImperativeMigration:
    """Wrapper for imperative migration functions."""

    def __init__(self, fromVersion: int, fn: Callable[[MigrationContext], None]) -> None:
        self.fromVersion = fromVersion
        self.fn = fn


def migration(fromVersion: int) -> Callable[[Callable[[MigrationContext], None]], _ImperativeMigration]:
    """Decorator for imperative migration functions.

    Usage::

        class Migrate:
            @migration(fromVersion=1)
            def from_v1(ctx: MigrationContext) -> None:
                ctx['new_field'] = ctx.pop('old_field') * 2
    """

    def decorator(fn: Callable[[MigrationContext], None]) -> _ImperativeMigration:
        return _ImperativeMigration(fromVersion, fn)

    return decorator


# ---------------------------------------------------------------------------
# Migration context (for imperative migrations)
# ---------------------------------------------------------------------------


class MigrationContext:
    """Dict-like wrapper for imperative migrations."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def pop(self, key: str, *default: Any) -> Any:
        return self._data.pop(key, *default)

    def drop(self, key: str) -> None:
        self._data.pop(key, None)

    def keys(self) -> Any:
        return self._data.keys()

    def toDict(self) -> dict[str, Any]:
        return dict(self._data)


# ---------------------------------------------------------------------------
# Migration resolution and application
# ---------------------------------------------------------------------------


def resolveMigrations(
    cls: type,
    fromVersion: int,
    toVersion: int,
) -> list[Migration | _ImperativeMigration]:
    """Collect and order migrations from *fromVersion* to *toVersion*.

    Walks the ``Migrate`` inner class on *cls*, collecting ``vN``
    attributes (declarative) and ``@migration``-decorated functions
    (imperative).  Migrations are ordered from oldest to newest.
    """
    migrateClass = getattr(cls, "Migrate", None)
    if migrateClass is None:
        raise MigrationError(
            f"{cls.__qualname__} version {toVersion} requires migration from "
            f"v{fromVersion}, but no Migrate class is defined."
        )

    # Collect all migrations with their version numbers
    migrations: dict[int, Migration | _ImperativeMigration] = {}

    for name in dir(migrateClass):
        attr = getattr(migrateClass, name)

        # Declarative: vN attributes
        if name.startswith("v") and name[1:].isdigit() and isinstance(attr, Migration):
            version = int(name[1:])
            migrations[version] = attr

        # Imperative: @migration decorated functions
        if isinstance(attr, _ImperativeMigration):
            migrations[attr.fromVersion] = attr

    # Build ordered chain: fromVersion → fromVersion+1 → ... → toVersion-1
    chain: list[Migration | _ImperativeMigration] = []
    for v in range(fromVersion, toVersion):
        if v not in migrations:
            raise MigrationError(
                f"No migration from v{v} to v{v + 1} defined on {cls.__qualname__}.Migrate (expected 'v{v}' attribute)."
            )
        chain.append(migrations[v])

    return chain


def applyMigrations(
    data: dict[str, Any],
    migrations: list[Migration | _ImperativeMigration],
    *,
    upgradeInPlace: bool = False,
) -> dict[str, Any]:
    """Apply a chain of migrations to *data*.

    Returns the migrated data dict.
    """
    result = dict(data)

    for mig in migrations:
        if isinstance(mig, _ImperativeMigration):
            ctx = MigrationContext(result)
            mig.fn(ctx)
            result = ctx.toDict()
        elif isinstance(mig, Migration):
            result = _applyDeclarativeMigration(result, mig, upgradeInPlace=upgradeInPlace)
        else:
            raise MigrationError(f"Unknown migration type: {type(mig)}")

    return result


def _applyDeclarativeMigration(
    data: dict[str, Any],
    mig: Migration,
    *,
    upgradeInPlace: bool = False,
) -> dict[str, Any]:
    """Apply a single declarative migration's operations."""
    result = dict(data)

    for op in mig.ops:
        if isinstance(op, _RenameOp):
            if op.old in result:
                result[op.new] = result.pop(op.old)

        elif isinstance(op, _AddOp):
            if op.field not in result:
                default = op.default
                if callable(default):
                    default = default()
                result[op.field] = default

        elif isinstance(op, _DropOp):
            result.pop(op.field, None)

        elif isinstance(op, _ConvertOp):
            if op.field in result:
                result[op.field] = op.via(result[op.field])

        elif isinstance(op, _DeriveOp):
            if op.fromField in result:
                result[op.field] = op.via(result[op.fromField])

        elif isinstance(op, _SplitOp):
            if op.field in result:
                source = result.pop(op.field)
                for targetField, fn in op.into.items():
                    result[targetField] = fn(source)

        elif isinstance(op, _MergeOp):
            values = [result.pop(f) for f in op.fields if f in result]
            if values:
                result[op.into] = op.via(*values)

        elif isinstance(op, _RequiresUpgradeOp) and not upgradeInPlace:
            raise UpgradeRequiredError(
                "This migration requires in-place file modification. "
                "Use upgradeInPlace=True or call upgradeFile() first."
            )

    return result
