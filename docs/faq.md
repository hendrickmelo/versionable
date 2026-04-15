# FAQ

## Is this production-ready?

Version 0.1.0 reflects a new *release*, not a new *idea*. The versioned-serialization pattern behind versionable has
been used in production C++ systems for over 15 years. From `CArchive`-based serialization through C++11 variadic
macros and modern template metaprogramming. Some version of this pattern has shipped in every project the authors have
worked on.

This Python implementation is new, but the design decisions are informed by hard-won experience:

- **Hash validated at class definition time** — catches schema drift at import, not at runtime. This is a deliberate
  architectural choice, not a shortcut.
- **`__init_subclass__`, not metaclass** — avoids metaclass conflicts and keeps inheritance simple.
- **Serialization names, not module paths** — hashes are stable across file moves and refactors.
- **Backends own their serialization** — each backend maps types to its native format (HDF5 uses datasets, not JSON
  inside HDF5). No lowest-common-denominator intermediate representation.

The test suite has a ~1:1 ratio of test code to source code (~4,400 lines of tests for ~4,500 lines of source), with
cross-backend round-trip coverage, migration chaining, HDF5 dtype preservation, lazy loading, and edge cases for union
types, enums, and nested versioned objects.

We use semver. The API is stable for documented use cases. Breaking changes will bump the minor version until 1.0.

## Is the schema hash required?

No. The hash parameter is entirely optional. This works fine:

```python
@dataclass
class Config(Versionable, version=1):
    name: str
    debug: bool = False
```

Without a hash, versionable still handles versioning, migrations, and serialization. You just won't get import-time
validation when your schema changes.

The hash exists to create *intentional friction*. When you change a field — rename it, change its type, add or remove
one — the hash forces you to acknowledge the change. You can't accidentally ship a schema change without writing a
migration. That friction is the feature: it moves data-format breakage from "silent corruption in production" to "error
during development."

Think of it like a type annotation or a lockfile hash: optional, zero runtime cost, catches mistakes before they
matter. If you're familiar with database migrations, it's the same idea — you'd never alter a production table without a
migration script. The hash enforces that same discipline for your data files. We recommend using it for any class whose
data files outlive a single session.

## Why does the base install require numpy?

It shouldn't — this is a known issue ([#17](https://github.com/hendrickmelo/versionable/issues/17)). A future release
will make numpy optional, required only when using ndarray fields or the HDF5 backend. The base install for config-only
workflows (JSON, TOML, YAML with scalars and strings) should have zero heavy dependencies. If this matters to you,
**please upvote the issue** — it helps us prioritize.

## Can I use this with Python 3.11 or earlier?

Not currently. Versionable requires Python 3.12+ and uses modern syntax features like the `type` statement and
`list[T]` / `dict[K, V]` built-in generics. There are no plans to backport to earlier Python versions.

## What happens if I load a file written by a newer version of my class?

You get a `VersionError`. Versionable supports forward migrations (old file → current code) but not downgrades. If your
file was written by version 5 of a class and your code defines version 3, loading will fail with a clear error message.

This is intentional — automatic downgrades risk data loss.

However, not every change requires a version bump. If you're adding a new field with a default value and it doesn't
affect older versions of your software, you can add the field without bumping the version — just update the hash. Older
files will load fine because the missing field falls back to its default. This way, you only break forwards compatibility when it's on purpose: a version bump means "older code can't read
these files," and the absence of one means "this change is safe to ignore."

Forwards incompatibility is the trade-off you make for simple migration code. Migrations only need to go in one
direction, which keeps them straightforward and composable. In practice, upgrading readers is far easier than
maintaining bidirectional migrations.

## How is versionable different from Pydantic or attrs?

Pydantic and attrs are validation and data-modeling libraries. They answer "is this data shaped correctly right now?"

Versionable answers a different question: "this data was shaped correctly *when it was written* — how do I load it now
that the code has changed?" It adds versioning, schema fingerprinting, and declarative migrations on top of standard
Python dataclasses. The two concerns are complementary — you could use Pydantic for API validation and versionable for
file persistence in the same project.

## Can I use my own file format?

Yes. Implement the `Backend` abstract class with `save()` and `load()` methods, then register it with
`registerBackend()`. See the [backends documentation](backends.md) for details.

## Why camelCase instead of snake_case?

This is a deliberate convention, not an oversight. The `Versionable` / `Serializable` pattern originated in C++ over 15
years ago, and the naming conventions carried forward intentionally. The Python implementation uses the same vocabulary
(`computeHash`, `registerConverter`, `skipDefaults`) that the authors have used across every language this pattern has
been implemented in.

We're aware this goes against PEP 8 convention for functions and variables. In practice, libraries like `unittest`
(`assertEqual`, `setUp`), `logging` (`getLogger`, `setLevel`), and Qt bindings (`paintEvent`, `setLayout`) use
camelCase throughout their public APIs and it hasn't blocked adoption. We chose consistency with the pattern's history
over convention conformance.

