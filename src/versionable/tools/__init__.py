"""Developer tooling that ships with versionable but is not part of its runtime API.

Nothing here is imported by the library itself, so a tool may depend on anything the library
declares, including its internals. Each tool is a module runnable with ``-m``::

    python -m versionable.tools.to_csharp mypkg.schemas

Nothing is re-exported from this package: importing a submodule here would make
``python -m versionable.tools.to_csharp`` import that submodule twice, once as a package
attribute and once as ``__main__``, which ``runpy`` warns about and which would run any
module-level state twice. Import the tool module directly to use one as a library.
"""

from __future__ import annotations
