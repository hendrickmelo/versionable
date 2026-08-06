using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Versionable.Converters;

/// <summary>
/// The converter set that ships with the library, and the module initializer that registers
/// it.
/// </summary>
/// <remarks>
/// Python counterpart: the block of <c>_registry.register(...)</c> calls at the top of
/// <c>src/versionable/_types.py</c>.
/// <para>
/// Registration order is resolution priority for the subclass matchers (first match wins), so
/// <see cref="All"/> is ordered, not a set. There is one matcher today — <c>Regex</c> — so
/// nothing currently depends on the order; a user converter registered later for a Regex
/// subtype would lose to the built-in, which is the pinned contract's behaviour and is why
/// the order is written down rather than left to reflection.
/// </para>
/// <para>
/// Enums are absent on purpose: see <see cref="EnumConverter"/>.
/// </para>
/// </remarks>
internal static class BuiltinConverters
{
    /// <summary>Every built-in converter, in registration order.</summary>
    internal static readonly IWireConverter[] All =
    [
        // Exact matches.
        new DateTimeConverter(),
        new DateTimeOffsetConverter(),
        new DateOnlyConverter(),
        new TimeOnlyConverter(),
        new TimeSpanConverter(),
        new GuidConverter(),
        new DecimalConverter(),
        new BytesConverter(),
        new ComplexConverter(),

        // One closed instantiation per dtype token with a C# element type (GRAMMAR §7).
        // complex64 is absent: .NET has no single-precision complex, so it can be read but
        // never declared.
        new TensorConverter<bool>(),
        new TensorConverter<sbyte>(),
        new TensorConverter<short>(),
        new TensorConverter<int>(),
        new TensorConverter<long>(),
        new TensorConverter<byte>(),
        new TensorConverter<ushort>(),
        new TensorConverter<uint>(),
        new TensorConverter<ulong>(),
        new TensorConverter<Half>(),
        new TensorConverter<float>(),
        new TensorConverter<double>(),
        new TensorConverter<Complex>(),

        new FilePathConverter(),

        // Subclass matchers, consulted in this order after every exact match has missed.
        new RegexConverter(),
    ];

    /// <summary>Registers <see cref="All"/> with <see cref="ConverterRegistry"/>.</summary>
    /// <remarks>
    /// Runs before any code in the assembly, so a consumer never has to remember to call it,
    /// and re-running it is harmless: registration replaces by type. It is separate from the
    /// initializer so tests that reset the registry can put it back.
    /// </remarks>
    internal static void RegisterAll()
    {
        foreach (IWireConverter converter in All)
        {
            ConverterRegistry.Register(converter);
        }
    }

    [ModuleInitializer]
    [SuppressMessage(
        "Usage",
        "CA2255:The 'ModuleInitializer' attribute should not be used in libraries",
        Justification = "The registration point is pinned by the contract (ADR-0003, and the "
            + "remarks on ConverterRegistry): the converter set must be in place before any "
            + "generated registration or user code runs, without a consumer having to call an "
            + "initializer. CA2255's concern is a library imposing start-up work on unrelated "
            + "code, and this initializer runs on first use of the Versionable assembly, which "
            + "is precisely when the registry is about to be read.")]
    internal static void Initialize() => RegisterAll();
}
