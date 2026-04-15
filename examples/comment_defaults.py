"""Example: commentDefaults with nested Versionable objects.

When saving with ``commentDefaults=True``, fields at their default value are
written as commented-out lines.  Section headers and metadata are always kept
so the file stays structurally valid — users uncomment only what they need.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import versionable
from versionable import Versionable


@dataclass
class DatabaseConfig(Versionable, version=1, name="DatabaseConfig"):
    host: str = "localhost"
    port: int = 5432
    maxConnections: int = 10
    timeoutSec: float = 30.0


@dataclass
class LoggingConfig(Versionable, version=1, name="LoggingConfig"):
    level: str = "INFO"
    filePath: str = "/var/log/app.log"
    rotateAfterMB: int = 100


@dataclass
class ServerConfig(Versionable, version=1, name="ServerConfig"):
    name: str = "my-server"
    debug: bool = False
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


if __name__ == "__main__":
    cfg = ServerConfig(
        database=DatabaseConfig(port=3306)
    )  # override default port to show how it appears in output files

    with tempfile.TemporaryDirectory() as tmp:
        toml_path = Path(tmp) / "config.toml"
        versionable.save(cfg, toml_path, commentDefaults=True)
        print("=== TOML ===\n")
        print(toml_path.read_text())

        yaml_path = Path(tmp) / "config.yaml"
        versionable.save(cfg, yaml_path, commentDefaults=True)
        print("=== YAML ===\n")
        print(yaml_path.read_text())

# Output TOML file:
#
# # name = "my-server"
# # debug = false
#
# [__versionable__]
# __OBJECT__ = "ServerConfig"
# __VERSION__ = 1
# __HASH__ = ""
#
# [database]
# __OBJECT__ = "DatabaseConfig"
# __VERSION__ = 1
# __HASH__ = ""
# # host = "localhost"
# port = 3306
# # maxConnections = 10
# # timeoutSec = 30.0
#
# [logging]
# __OBJECT__ = "LoggingConfig"
# __VERSION__ = 1
# __HASH__ = ""
# # level = "INFO"
# # filePath = "/var/log/app.log"
# # rotateAfterMB = 100
#
#
# Output YAML file:
#
# # name: my-server
# # debug: false
# database:
#   __OBJECT__: DatabaseConfig
#   __VERSION__: 1
#   __HASH__: ''
#   host: localhost
#   port: 3306
#   maxConnections: 10
#   timeoutSec: 30.0
# logging:
#   __OBJECT__: LoggingConfig
#   __VERSION__: 1
#   __HASH__: ''
# #   level: INFO
# #   filePath: /var/log/app.log
# #   rotateAfterMB: 100
# __versionable__:
#   __OBJECT__: ServerConfig
#   __VERSION__: 1
#   __HASH__: ''
