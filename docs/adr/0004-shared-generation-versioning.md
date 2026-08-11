# Package versions share a generation (major.minor) across languages

The PyPI `versionable` and NuGet `Versionable` packages share their `major.minor` — the "generation". Same generation ⇒
same canonical type grammar, interchangeable files, feature equivalence (minus documented gaps, e.g. the deferred HDF5
session API). Minor bumps are joint monorepo releases (an implementation with no changes ships an empty bump); the patch
digit is per-package so analyzer/backend fixes on one side don't force empty releases on the other. Python tags stay
`vX.Y.Z`; C# releases tag `dotnet-vX.Y.Z`. **Within a generation the Python release lands first and the `dotnet-vX.Y.Z`
tag follows**, so the shared `major.minor` is already visible in `pyproject.toml` when the NuGet package is built —
`publish-nuget.yml` checks exactly that and refuses a tag whose generation has drifted. Fully independent streams were
rejected because the user-facing compatibility rule ("match the minor version and files interchange") is worth more than
release-cadence freedom; full lockstep including patch was rejected because early C#-only fixes would spam PyPI with
empty releases.

The C# implementation debuts at the generation carrying the grammar cleanup (bare serialization names,
`ndarray[<dtype>]`, `Literal` quoting) — one breaking release covers both languages.
