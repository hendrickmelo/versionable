namespace Versionable.Converters;

/// <summary>
/// Converts one CLR type to and from its wire representation.
/// </summary>
/// <remarks>
/// Python counterpart: <c>_TypeConverter</c> and the <c>registerConverter()</c> entry point
/// in <c>src/versionable/_types.py</c>, whose converters are a
/// <c>(serialize, deserialize)</c> pair plus <c>matchSubclasses</c> and a serialization
/// name.
/// <para>
/// The wire representations are fixed by cross-language interchange, not by C# taste:
/// <c>Guid</c> as a lowercase hyphenated UUID string, <c>decimal</c> as a string,
/// <c>byte[]</c> as base64, <c>Complex</c> as <c>[re, im]</c>, <c>TimeSpan</c> as total
/// seconds (<c>docs/plans/csharp-port.md</c>, § "Wire format / converters").
/// </para>
/// </remarks>
public interface IWireConverter
{
    /// <summary>The CLR type this converter handles.</summary>
    Type ClrType { get; }

    /// <summary>
    /// Name <see cref="ClrType"/> canonicalises to in schema hashes — a bare name such as
    /// <c>datetime</c>, <c>UUID</c>, or <c>Path</c>, matching what Python emits (ADR-0001).
    /// Python counterpart: the <c>name=</c> argument of <c>registerConverter()</c>.
    /// </summary>
    string SerializationName { get; }

    /// <summary>
    /// Whether subclasses of <see cref="ClrType"/> also route through this converter.
    /// Python counterpart: <c>matchSubclasses</c>.
    /// </summary>
    bool MatchSubclasses { get; }

    /// <summary>Converts a value to its wire representation.</summary>
    /// <param name="value">The CLR value; never <see langword="null"/> (nulls short-circuit upstream).</param>
    /// <returns>A primitive, list, or dictionary the backends can write.</returns>
    /// <exception cref="Errors.ConverterException">The value cannot be represented on the wire.</exception>
    object? ToWire(object value);

    /// <summary>Converts a wire representation back to a CLR value.</summary>
    /// <param name="wireValue">The value as read from the file.</param>
    /// <param name="targetType">
    /// The declared field type, which may be a subclass of <see cref="ClrType"/> when
    /// <see cref="MatchSubclasses"/> is set. Python counterpart: the second argument of the
    /// <c>deserialize</c> lambda.
    /// </param>
    /// <returns>The reconstructed CLR value.</returns>
    /// <exception cref="Errors.ConverterException">The wire value is not valid for <paramref name="targetType"/>.</exception>
    object FromWire(object wireValue, Type targetType);
}
