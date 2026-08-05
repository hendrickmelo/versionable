using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;

namespace Versionable.Converters;

/// <summary>
/// Resolves the <see cref="IWireConverter"/> for a CLR type: exact match first, then the
/// registered subclass matchers in registration order.
/// </summary>
/// <remarks>
/// Python counterpart: <c>ConverterRegistry</c> and its module-level <c>_registry</c>
/// singleton in <c>src/versionable/_types.py</c>. Built-in converters register from a
/// <c>[ModuleInitializer]</c>; user converters call <see cref="Register"/> directly, the
/// analogue of Python's <c>registerConverter()</c>.
/// <para>
/// Thread-safe for concurrent registration and lookup. Resolution results are deliberately
/// <em>not</em> cached: a cache would have to be invalidated on every registration, and an
/// invalidation racing a concurrent fill can leave a stale entry that outlives the
/// registration that should have replaced it. Both lookups on the resolve path are cheap — an
/// <see cref="ConcurrentDictionary{TKey, TValue}"/> hit, then a walk of a handful of subclass
/// matchers — so there is nothing to buy back.
/// </para>
/// </remarks>
public static class ConverterRegistry
{
    private static readonly ConcurrentDictionary<Type, IWireConverter> _exact = new();
    private static readonly object _registrationLock = new();

    // Replaced wholesale under _registrationLock, read without any lock. Readers take one
    // reference and walk it; a concurrent registration swaps in a new array and never mutates
    // the one already handed out, so there is nothing to tear.
    //
    // volatile covers the other half — publication. Without it, a weakly-ordered architecture
    // (arm64, which is every Apple silicon dev machine and an increasing share of CI) may let
    // a reader observe the new array reference before the writes that filled its elements are
    // visible, handing back a array of nulls. The volatile write on assignment and volatile
    // read on access order the two.
    private static volatile IWireConverter[] _subclassMatchers = [];

    /// <summary>Registers <paramref name="converter"/>, replacing any converter for the same type.</summary>
    /// <param name="converter">The converter to register.</param>
    public static void Register(IWireConverter converter)
    {
        ArgumentNullException.ThrowIfNull(converter);

        if (converter.MatchSubclasses)
        {
            lock (_registrationLock)
            {
                _subclassMatchers = [.. _subclassMatchers, converter];
            }
        }
        else
        {
            _exact[converter.ClrType] = converter;
        }
    }

    /// <summary>Finds the converter for <paramref name="type"/>.</summary>
    /// <param name="type">A declared field type.</param>
    /// <param name="converter">The matching converter, when one is registered.</param>
    /// <returns><see langword="true"/> when a converter handles <paramref name="type"/>.</returns>
    public static bool TryResolve(Type type, [NotNullWhen(true)] out IWireConverter? converter)
    {
        if (_exact.TryGetValue(type, out converter))
        {
            return true;
        }

        foreach (IWireConverter candidate in _subclassMatchers)
        {
            if (candidate.ClrType.IsAssignableFrom(type))
            {
                converter = candidate;
                return true;
            }
        }

        converter = null;
        return false;
    }

    /// <summary>Empties the registry. Test seam; there is no public unregister.</summary>
    internal static void Reset()
    {
        lock (_registrationLock)
        {
            _exact.Clear();
            _subclassMatchers = [];
        }
    }
}
