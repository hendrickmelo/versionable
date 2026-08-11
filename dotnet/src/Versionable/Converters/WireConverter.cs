using System.Globalization;
using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// Base for the built-in converters: pins <see cref="ClrType"/> to <typeparamref name="T"/>
/// and turns the interface's <see cref="object"/> arguments into typed ones.
/// </summary>
/// <typeparam name="T">The CLR type the converter handles.</typeparam>
/// <remarks>
/// Every "the value is not what this converter handles" failure produces the same shaped
/// message from one place, which is the only reason the base exists — the conversions
/// themselves are one-liners.
/// </remarks>
internal abstract class WireConverter<T> : IWireConverter
{
    /// <inheritdoc/>
    public Type ClrType => typeof(T);

    /// <inheritdoc/>
    public abstract string SerializationName { get; }

    /// <inheritdoc/>
    public virtual bool MatchSubclasses => false;

    /// <inheritdoc/>
    public object? ToWire(object value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (value is not T typed)
        {
            throw new ConverterException(
                $"{GetType().Name} serializes {typeof(T).Name}, not {value.GetType().Name}.");
        }

        return ToWireCore(typed);
    }

    /// <inheritdoc/>
    public object FromWire(object wireValue, Type targetType)
    {
        ArgumentNullException.ThrowIfNull(wireValue);
        ArgumentNullException.ThrowIfNull(targetType);
        return FromWireCore(wireValue, targetType);
    }

    /// <summary>Converts a typed value to its wire representation.</summary>
    /// <param name="value">The value, never <see langword="null"/>.</param>
    /// <returns>The wire representation.</returns>
    protected abstract object? ToWireCore(T value);

    /// <summary>Converts a wire representation back to <typeparamref name="T"/>.</summary>
    /// <param name="wireValue">The value as read from the file, never <see langword="null"/>.</param>
    /// <param name="targetType">The declared field type.</param>
    /// <returns>The reconstructed value.</returns>
    protected abstract object FromWireCore(object wireValue, Type targetType);

    /// <summary>Narrows a wire value to <see cref="string"/>, or explains why it is not one.</summary>
    /// <param name="wireValue">The value as read from the file.</param>
    /// <returns>The string.</returns>
    /// <exception cref="ConverterException">The wire value is not a string.</exception>
    protected string RequireString(object wireValue)
    {
        if (wireValue is string text)
        {
            return text;
        }

        throw new ConverterException(
            $"{SerializationName} is written as a string; the file holds "
            + $"{wireValue.GetType().Name} ({FormatForMessage(wireValue)}).");
    }

    /// <summary>Narrows a wire value to <see cref="double"/>, accepting any numeric form.</summary>
    /// <param name="wireValue">The value as read from the file.</param>
    /// <returns>The number.</returns>
    /// <exception cref="ConverterException">The wire value is not numeric.</exception>
    /// <remarks>
    /// Backends decode numbers to whichever CLR type their parser picks — a TOML integer
    /// arrives as <see cref="long"/> where the same value in JSON may arrive as
    /// <see cref="double"/> — so the accepted set is "anything convertible", not one type.
    /// </remarks>
    protected double RequireDouble(object wireValue)
    {
        if (wireValue is IConvertible convertible and not string and not bool)
        {
            try
            {
                return convertible.ToDouble(CultureInfo.InvariantCulture);
            }
            catch (Exception e) when (e is FormatException or OverflowException or InvalidCastException)
            {
                throw new ConverterException(
                    $"{SerializationName} is written as a number; the file holds "
                    + $"{FormatForMessage(wireValue)}.",
                    e);
            }
        }

        throw new ConverterException(
            $"{SerializationName} is written as a number; the file holds "
            + $"{wireValue.GetType().Name} ({FormatForMessage(wireValue)}).");
    }

    private static string FormatForMessage(object value) =>
        value is IFormattable formattable
            ? formattable.ToString(null, CultureInfo.InvariantCulture)
            : value.ToString() ?? "null";
}
