using System.Collections;
using System.Globalization;
using System.Numerics;
using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// <see cref="Complex"/> ⇄ a two-element <c>[real, imaginary]</c> list.
/// </summary>
/// <remarks>
/// Python counterpart: <c>registerConverter(complex, lambda v: [v.real, v.imag], ...)</c> in
/// <c>src/versionable/_types.py</c>. The wire value is a list of two CLR
/// <see cref="double"/>s; how a backend renders that list — a JSON array, a YAML sequence, a
/// TOML array — is the backend's business.
/// </remarks>
internal sealed class ComplexConverter : WireConverter<Complex>
{
    /// <inheritdoc/>
    public override string SerializationName => "complex";

    /// <inheritdoc/>
    protected override object ToWireCore(Complex value) => new List<object?> { value.Real, value.Imaginary };

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType)
    {
        if (wireValue is not IEnumerable sequence || wireValue is string)
        {
            throw new ConverterException(
                $"complex is written as a two-element [real, imaginary] list; the file holds "
                + $"{wireValue.GetType().Name}.");
        }

        List<double> parts = [];
        foreach (object? part in sequence)
        {
            if (parts.Count == 2)
            {
                throw new ConverterException("complex needs exactly two elements, [real, imaginary].");
            }

            if (part is not IConvertible convertible || part is string || part is bool)
            {
                throw new ConverterException(
                    $"complex components must be numbers; found {part?.GetType().Name ?? "null"}.");
            }

            parts.Add(convertible.ToDouble(CultureInfo.InvariantCulture));
        }

        if (parts.Count != 2)
        {
            throw new ConverterException(
                $"complex needs exactly two elements, [real, imaginary]; found {parts.Count}.");
        }

        return new Complex(parts[0], parts[1]);
    }
}
