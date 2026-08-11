using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Reflection;
using Versionable.Engine;
using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// Enum ⇄ its member value: the number a member declares, or the string
/// <see cref="EnumValueAttribute"/> gives it.
/// </summary>
/// <remarks>
/// Python counterparts: the <c>isinstance(value, Enum)</c> arm of <c>_serializeTyped</c> and
/// <c>_deserializeEnum</c> in <c>src/versionable/_types.py</c>.
/// <para>
/// <strong>Not an <see cref="IWireConverter"/>, and deliberately not in
/// <see cref="ConverterRegistry"/>.</strong> A converter carries one Serialization Name, and
/// enums do not share one: each hashes as its own bare type name (GRAMMAR §9), so a
/// registered <c>typeof(Enum)</c> matcher would have to answer <c>SerializationName</c> with
/// something wrong for every enum in the program. Python draws the same line — its registry
/// holds no enum entry, and <c>serialize</c>/<c>deserialize</c> dispatch enums on a branch of
/// their own after the registry lookup misses. The engine calls this class from that branch.
/// </para>
/// <para>
/// Member values are not hash-significant: adding a member, removing one, or changing the
/// value of one never changes a schema hash, so an unknown value on the wire is a routine
/// version skew rather than a corrupt file. That is what
/// <see cref="EnumFallbackAttribute"/> is for.
/// </para>
/// </remarks>
internal static class EnumConverter
{
    private static readonly ConcurrentDictionary<Type, EnumWireMap> _maps = new();

    /// <summary>Converts an enum member to its wire value.</summary>
    /// <param name="value">The member.</param>
    /// <returns>
    /// The <see cref="EnumValueAttribute"/> string when the member declares one, otherwise the
    /// member's numeric value boxed as the enum's underlying type.
    /// </returns>
    /// <remarks>
    /// The <see cref="Type"/> reached through <see cref="object.GetType"/> needs no
    /// <see cref="DynamicallyAccessedMembersAttribute"/>: the trimmer models it as fully
    /// annotated, since the instance it came from proves the type is live.
    /// </remarks>
    internal static object ToWire(Enum value)
    {
        ArgumentNullException.ThrowIfNull(value);
        Type enumType = value.GetType();
        EnumWireMap map = MapFor(enumType);

        string? name = Enum.GetName(enumType, value);
        if (name is not null && map.WireStringByName.TryGetValue(name, out string? wireString))
        {
            return wireString;
        }

        // Unnamed values (a cast integer, a flags combination) fall through to the number,
        // which is the only thing that can round-trip them.
        return Convert.ChangeType(value, map.UnderlyingType, CultureInfo.InvariantCulture);
    }

    /// <summary>Converts a wire value back to a member of <paramref name="enumType"/>.</summary>
    /// <param name="wireValue">The value as read from the file.</param>
    /// <param name="enumType">The declared enum type.</param>
    /// <returns>The matching member, or the <see cref="EnumFallbackAttribute"/> member.</returns>
    /// <exception cref="ConverterException">
    /// No member matches and the enum declares no fallback.
    /// </exception>
    internal static object FromWire(
        object wireValue,
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields)] Type enumType)
    {
        ArgumentNullException.ThrowIfNull(wireValue);
        ArgumentNullException.ThrowIfNull(enumType);
        if (!enumType.IsEnum)
        {
            throw new ConverterException($"{enumType.Name} is not an enum type.");
        }

        EnumWireMap map = MapFor(enumType);

        if (wireValue is string text)
        {
            if (map.MemberByWireString.TryGetValue(text, out object? member))
            {
                return member;
            }
        }
        else if (TryToUnderlying(wireValue, map.UnderlyingType, out object? underlying)
            && Enum.IsDefined(enumType, underlying))
        {
            return Enum.ToObject(enumType, underlying);
        }

        if (map.Fallback is not null)
        {
            VersionableLog.Warn(
                $"Unknown {enumType.Name} value {wireValue}, using fallback {map.Fallback}");
            return map.Fallback;
        }

        throw new ConverterException(
            $"Unknown {enumType.Name} value: {wireValue}. Mark a member with [EnumFallback] to "
            + "accept values written by a newer version of the schema.");
    }

    private static EnumWireMap MapFor(
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields)] Type enumType)
    {
        // Built then published rather than via GetOrAdd: the factory delegate would drop the
        // DynamicallyAccessedMembers annotation on enumType and trip the trim analyzer. A race
        // between two builders costs one redundant reflection pass and nothing else, since the
        // maps are equal by construction.
        if (_maps.TryGetValue(enumType, out EnumWireMap? cached))
        {
            return cached;
        }

        EnumWireMap built = EnumWireMap.Build(enumType);
        _maps[enumType] = built;
        return built;
    }

    private static bool TryToUnderlying(object wireValue, Type underlyingType, out object underlying)
    {
        underlying = 0;
        if (wireValue is not IConvertible convertible || wireValue is bool)
        {
            return false;
        }

        try
        {
            underlying = convertible.ToType(underlyingType, CultureInfo.InvariantCulture);
            return true;
        }
        catch (Exception e) when (e is FormatException or OverflowException or InvalidCastException)
        {
            return false;
        }
    }

    /// <summary>The reflected wire mapping for one enum type.</summary>
    private sealed class EnumWireMap
    {
        private EnumWireMap(
            Type underlyingType,
            Dictionary<string, string> wireStringByName,
            Dictionary<string, object> memberByWireString,
            object? fallback)
        {
            UnderlyingType = underlyingType;
            WireStringByName = wireStringByName;
            MemberByWireString = memberByWireString;
            Fallback = fallback;
        }

        internal Type UnderlyingType { get; }

        /// <summary>Member name to its <see cref="EnumValueAttribute"/> string.</summary>
        internal Dictionary<string, string> WireStringByName { get; }

        /// <summary><see cref="EnumValueAttribute"/> string to the boxed member.</summary>
        internal Dictionary<string, object> MemberByWireString { get; }

        /// <summary>The boxed <see cref="EnumFallbackAttribute"/> member, if any.</summary>
        internal object? Fallback { get; }

        internal static EnumWireMap Build(
            [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields)] Type enumType)
        {
            Dictionary<string, string> wireStringByName = [];
            Dictionary<string, object> memberByWireString = new(StringComparer.Ordinal);
            object? fallback = null;

            foreach (string name in Enum.GetNames(enumType))
            {
                FieldInfo field = enumType.GetField(name, BindingFlags.Public | BindingFlags.Static)
                    ?? throw new ConverterException($"Enum {enumType.Name} has no field named {name}.");
                object member = Enum.Parse(enumType, name);

                if (field.GetCustomAttribute<EnumValueAttribute>() is { } enumValue)
                {
                    wireStringByName[name] = enumValue.Value;
                    if (!memberByWireString.TryAdd(enumValue.Value, member))
                    {
                        throw new ConverterException(
                            $"Enum {enumType.Name} has two members with [EnumValue(\"{enumValue.Value}\")]. "
                            + "Wire values must be unique or a file cannot be read back unambiguously.");
                    }
                }

                if (field.GetCustomAttribute<EnumFallbackAttribute>() is not null)
                {
                    if (fallback is not null)
                    {
                        throw new ConverterException(
                            $"Enum {enumType.Name} marks more than one member [EnumFallback]; at most "
                            + "one member may be the fallback.");
                    }

                    fallback = member;
                }
            }

            return new EnumWireMap(
                Enum.GetUnderlyingType(enumType),
                wireStringByName,
                memberByWireString,
                fallback);
        }
    }
}
