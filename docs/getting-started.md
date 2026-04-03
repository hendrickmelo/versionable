# Getting Started

## Installation

```bash
pip install git+https://github.com/hendrickmelo/versionable.git
```

With HDF5 support (needed for saving large arrays to `.h5` files — [see why](backends.md#hdf5)):

```bash
pip install "versionable[hdf5] @ git+https://github.com/hendrickmelo/versionable.git"
```

## The Hash

### Why it exists

Serialized files outlive your code. If you rename a field, change a type, or add a required field, files saved by the
old version of your class will silently load with wrong values — or worse, load without error while corrupting
downstream logic.

The `hash` parameter is a 6-character fingerprint of your class's field names and types. versionable recomputes it every
time your module is imported and raises `HashMismatchError` immediately if the class definition no longer matches what
was declared. You find out at startup, not buried in a production bug.

### Computing the hash for a new class

1. Define your class without a hash:

   ```python
   @dataclass
   class MyConfig(Versionable, version=1):
       name: str
       value: float
   ```

2. Run your code — validation is skipped when `hash` is absent or empty.

3. Print the hash:

   ```python
   print(MyConfig.hash())
   # e.g. "4b7866"
   ```

4. Add it to the class definition:

   ```python
   @dataclass
   class MyConfig(Versionable, version=1, hash="4b7866"):
       name: str
       value: float
   ```

### What happens when the schema changes

Suppose you saved a file with the original class:

```python
# v1 — field is called "value"
@dataclass
class MyConfig(Versionable, version=1, hash="4b7866"):
    name: str
    value: float
```

The YAML file on disk looks like:

```yaml
name: experiment-A
value: 9.81
__meta__:
  __OBJECT__: MyConfig
  __VERSION__: 1
  __HASH__: 4b7866
```

Now you rename the field without updating the hash:

```python
@dataclass
class MyConfig(Versionable, version=1, hash="4b7866"):  # ← hash still matches v1, but fields changed!
    name: str
    magnitude: float          # renamed from "value"
```

Without hash validation, `versionable.load(MyConfig, "config.yaml")` would silently return an object where `magnitude`
is unset (or a default), and the `value = 9.81` from the file would be silently discarded. versionable catches this
before you ever call `load`:

```text
HashMismatchError: MyConfig — declared hash '4b7866' does not match computed 'a70249'
```

That's your signal to recompute the hash, bump the version, and add a migration so existing files can be upgraded.
Here's the correct v2:

```python
from versionable import Versionable, Migration

@dataclass
class MyConfig(Versionable, version=2, hash="a70249"):
    name: str
    magnitude: float          # renamed from "value"

    class Migrate:
        v1 = Migration().rename("value", "magnitude")
```

Now `versionable.load(MyConfig, "config.yaml")` reads the old `value: 9.81` from the file, renames it to `magnitude` on
the fly, and returns a fully populated `MyConfig` instance.

See [Migrations](migrations.md) for the full range of migration operations.

## Dev Mode

During rapid iteration you can suppress hash errors so mismatches are warnings instead:

```python
from versionable import ignoreHashErrors
ignoreHashErrors(True)
```

Turn it off before committing. (or submit a PR for [#2](https://github.com/hendrickmelo/versionable/issues/2))
