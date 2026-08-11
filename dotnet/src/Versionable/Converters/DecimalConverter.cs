using System.Globalization;
using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// <see cref="decimal"/> ⇄ a decimal string.
/// </summary>
/// <remarks>
/// Python counterpart: <c>registerConverter(Decimal, str, Decimal)</c> in
/// <c>src/versionable/_types.py</c>. A string, not a JSON number, so no digit is lost to a
/// double on the way through — which is the whole point of declaring the field
/// <see cref="decimal"/>.
/// <para>
/// <strong>Python's <c>Decimal</c> is wider than <see cref="decimal"/>.</strong> Its
/// coefficient and exponent are unbounded, and it has <c>NaN</c> and <c>Infinity</c> values;
/// .NET's is a fixed 96-bit coefficient with a scale of 0–28 and no non-finite values. A file
/// holding <c>1E+50</c>, <c>NaN</c>, or thirty significant digits therefore loads in Python
/// and raises <see cref="ConverterException"/> here. That is the documented failure mode: the
/// alternative is silently rounding or saturating a value the schema promised was exact.
/// </para>
/// </remarks>
internal sealed class DecimalConverter : WireConverter<decimal>
{
    /// <inheritdoc/>
    public override string SerializationName => "Decimal";

    /// <inheritdoc/>
    protected override object ToWireCore(decimal value) => value.ToString(CultureInfo.InvariantCulture);

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType)
    {
        string text = RequireString(wireValue);

        // NumberStyles.Float admits the exponent notation Python's Decimal round-trips
        // through ('1E+7'); AllowThousands is deliberately excluded, since no writer emits
        // group separators and accepting them would make ',' locale-dependent.
        if (!decimal.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out decimal result))
        {
            throw new ConverterException(
                $"'{text}' is not a decimal .NET can represent. Python's Decimal is unbounded and "
                + "has NaN/Infinity; System.Decimal holds 28-29 significant digits in the range "
                + "±7.9228E+28 and nothing else.");
        }

        return result;
    }
}
