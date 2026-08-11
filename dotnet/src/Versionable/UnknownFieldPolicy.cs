namespace Versionable;

/// <summary>
/// How to treat fields present in a file but not declared on the target type.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>unknown</c> string parameter (<c>"ignore" | "error" |
/// "preserve"</c>) on <c>Versionable.__init_subclass__</c> in
/// <c>src/versionable/_base.py</c>.
/// </remarks>
public enum UnknownFieldPolicy
{
    /// <summary>Drop unknown fields silently. Python's <c>"ignore"</c> (the default).</summary>
    Ignore = 0,

    /// <summary>Throw <see cref="Errors.UnknownFieldException"/>. Python's <c>"error"</c>.</summary>
    Error = 1,

    /// <summary>Keep unknown fields so a later save round-trips them. Python's <c>"preserve"</c>.</summary>
    Preserve = 2,
}
