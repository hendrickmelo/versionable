namespace Versionable;

/// <summary>
/// Marks the enum member to fall back to when a file holds a value this enum does not
/// define.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>VERSIONABLE_FALLBACK</c> class attribute assigned after the
/// enum body, read by <c>_deserializeEnum</c> in <c>src/versionable/_types.py</c>. The
/// substitution is logged as a warning, as it is in Python.
/// <para>
/// At most one member per enum may carry it. Without it, an unknown value is a
/// <see cref="Errors.ConverterException"/> — which is the right default when silently
/// collapsing unknown values would hide a version skew.
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Field, Inherited = false)]
public sealed class EnumFallbackAttribute : Attribute;
