using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// <see cref="byte"/><c>[]</c> ⇄ a base64 string.
/// </summary>
/// <remarks>
/// Python counterpart: <c>registerConverter(bytes, base64.b64encode, base64.b64decode)</c> in
/// <c>src/versionable/_types.py</c> — the standard alphabet, padded, which is what
/// <see cref="Convert.ToBase64String(byte[])"/> emits.
/// <para>
/// <c>byte[]</c> is the grammar's one array carve-out: it renders <c>bytes</c>, not
/// <c>ndarray[uint8]</c> (GRAMMAR §7). A field that really is an array of small unsigned
/// integers should be declared <c>Tensor&lt;byte&gt;</c>.
/// </para>
/// </remarks>
internal sealed class BytesConverter : WireConverter<byte[]>
{
    /// <inheritdoc/>
    public override string SerializationName => "bytes";

    /// <inheritdoc/>
    protected override object ToWireCore(byte[] value) => Convert.ToBase64String(value);

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType)
    {
        // The HDF5 backend hands opaque byte data back natively rather than base64-encoded.
        if (wireValue is byte[] raw)
        {
            return raw;
        }

        string text = RequireString(wireValue);
        try
        {
            return Convert.FromBase64String(text);
        }
        catch (FormatException e)
        {
            throw new ConverterException("Value is not valid base64.", e);
        }
    }
}
