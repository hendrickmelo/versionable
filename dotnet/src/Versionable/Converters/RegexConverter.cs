using System.Text.RegularExpressions;
using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// <see cref="Regex"/> ⇄ its pattern string.
/// </summary>
/// <remarks>
/// Python counterpart: <c>registerConverter(re.Pattern, lambda v: v.pattern, re.compile,
/// matchSubclasses=True)</c> in <c>src/versionable/_types.py</c>.
/// <para>
/// <strong>Options are dropped, in both directions.</strong> Only the pattern reaches the
/// wire, so a <see cref="RegexOptions.IgnoreCase"/> regex comes back case-sensitive —
/// exactly as a Python <c>re.IGNORECASE</c> pattern comes back without its flag. This is a
/// pre-existing quirk of the Python format, kept rather than fixed so the two implementations
/// read each other's files; encoding the flags would need a wire change on both sides.
/// Inline modifiers written into the pattern itself (<c>(?i)</c>) do survive, since they are
/// part of the pattern text.
/// </para>
/// <para>
/// A subclass-matched field is reconstructed as a plain <see cref="Regex"/>, not as the
/// declared subclass: nothing on the wire says how to build one. Python behaves the same way
/// — <c>re.compile</c> always returns a <c>re.Pattern</c>.
/// </para>
/// </remarks>
internal sealed class RegexConverter : WireConverter<Regex>
{
    /// <inheritdoc/>
    public override string SerializationName => "Pattern";

    /// <inheritdoc/>
    public override bool MatchSubclasses => true;

    /// <inheritdoc/>
    protected override object ToWireCore(Regex value) => value.ToString();

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType)
    {
        string pattern = RequireString(wireValue);
        try
        {
            return new Regex(pattern);
        }
        catch (ArgumentException e)
        {
            throw new ConverterException(
                $"'{pattern}' is not a valid .NET regular expression. Python and .NET regex "
                + "syntaxes overlap but are not identical.",
                e);
        }
    }
}
