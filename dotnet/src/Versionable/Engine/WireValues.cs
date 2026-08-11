using System.Collections;
using System.Collections.Frozen;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.CompilerServices;
using Versionable.Converters;
using Versionable.Errors;

namespace Versionable.Engine;

/// <summary>
/// The serialization walker: CLR values down to the primitives, lists, and dictionaries a
/// backend can write, and back up again.
/// </summary>
/// <remarks>
/// Python counterpart: <c>serialize</c> / <c>deserialize</c> and their helpers in
/// <c>src/versionable/_types.py</c>. One engine serves every text backend (ADR-0003); backends
/// call in from their <c>Save</c> and <c>Load</c>, which is why the walker takes the backend's
/// <see cref="Backends.IVersionableBackend.NativeTypes"/> rather than assuming JSON.
/// <para>
/// <b>Write dispatches on the runtime type of the value, read dispatches on the declared type of
/// the field.</b> That asymmetry is Python's and it is not an accident: writing needs to know
/// what a value <em>is</em> (so a subclass in a base-typed collection writes its own envelope),
/// while reading needs to know what it must <em>become</em>. It is also what makes reading the
/// side that needs generated help — see <see cref="Read(object?, Type)"/>.
/// </para>
/// </remarks>
public static class WireValues
{
    [ThreadStatic]
    private static WriteScope? _writeScope;

    [ThreadStatic]
    private static ReadScope? _readScope;

    private static readonly FrozenSet<Type> _passThroughTypes = new[]
    {
        typeof(string), typeof(bool),
        typeof(sbyte), typeof(byte), typeof(short), typeof(ushort),
        typeof(int), typeof(uint), typeof(long), typeof(ulong),
        typeof(float), typeof(double),
    }.ToFrozenSet();

    // ------------------------------------------------------------------
    // Write
    // ------------------------------------------------------------------

    /// <summary>
    /// Lowers <paramref name="value"/> to its wire form, joining the walk already in progress.
    /// </summary>
    /// <remarks>
    /// This is the overload generated <see cref="FieldDescriptor.WireWriter"/> delegates call for
    /// their elements: it picks up the ambient backend native types and cycle-detection set
    /// instead of starting a fresh walk.
    /// </remarks>
    /// <param name="value">The CLR value to lower.</param>
    /// <returns>A primitive, list, or dictionary the backend can write.</returns>
    public static object? Write(object? value) => Write(value, nativeTypes: null);

    /// <summary>Lowers <paramref name="value"/> to its wire form.</summary>
    /// <param name="value">The CLR value to lower.</param>
    /// <param name="nativeTypes">
    /// CLR types the backend stores natively and that must therefore be passed through untouched.
    /// Ignored when a walk is already in progress — the outermost call owns the walk.
    /// </param>
    /// <returns>A primitive, list, or dictionary the backend can write.</returns>
    /// <exception cref="UnsupportedTypeException">Nothing knows how to lower the value's type.</exception>
    /// <exception cref="CircularReferenceException">The object graph contains a cycle.</exception>
    public static object? Write(object? value, IReadOnlySet<Type>? nativeTypes)
    {
        if (_writeScope is not null)
        {
            return WriteValue(value, _writeScope);
        }

        WriteScope scope = new(nativeTypes ?? FrozenSet<Type>.Empty);
        _writeScope = scope;
        try
        {
            return WriteValue(value, scope);
        }
        finally
        {
            _writeScope = null;
        }
    }

    /// <summary>
    /// Lowers a whole object's raw field values, in declaration order.
    /// </summary>
    /// <remarks>
    /// What a text backend calls from its <c>Save</c>. The root envelope is not included: the
    /// backend receives it separately and writes it in whatever way its format wants. Nested
    /// objects <em>do</em> come back enveloped, because nothing downstream of here knows what
    /// type they were.
    /// </remarks>
    /// <param name="rawFields">Raw values keyed by wire name, as handed to the backend.</param>
    /// <param name="metadata">Metadata of the type being written; supplies order and writers.</param>
    /// <param name="nativeTypes">CLR types the backend stores natively.</param>
    /// <returns>Wire values keyed by wire name, in <see cref="VersionableMetadata.Fields"/> order.</returns>
    public static IReadOnlyDictionary<string, object?> WriteFields(
        IReadOnlyDictionary<string, object?> rawFields,
        VersionableMetadata metadata,
        IReadOnlySet<Type>? nativeTypes)
    {
        ArgumentNullException.ThrowIfNull(rawFields);
        ArgumentNullException.ThrowIfNull(metadata);

        WriteScope scope = _writeScope ?? new WriteScope(nativeTypes ?? FrozenSet<Type>.Empty);
        bool ownsScope = _writeScope is null;
        _writeScope = scope;
        try
        {
            Dictionary<string, object?> wire = new(rawFields.Count, StringComparer.Ordinal);
            foreach (FieldDescriptor field in metadata.Fields)
            {
                if (rawFields.TryGetValue(field.WireName, out object? raw))
                {
                    scope.Path = field.WireName;
                    wire[field.WireName] = WriteField(field, raw, scope);
                }
            }

            // Keys the type does not declare: an `unknown = Preserve` load put them there, and
            // dropping them here would make preserve mean "preserve until the next save".
            foreach (KeyValuePair<string, object?> extra in rawFields)
            {
                if (!wire.ContainsKey(extra.Key))
                {
                    scope.Path = extra.Key;
                    wire[extra.Key] = WriteValue(extra.Value, scope);
                }
            }

            scope.Path = string.Empty;
            return wire;
        }
        finally
        {
            if (ownsScope)
            {
                _writeScope = null;
            }
        }
    }

    private static object? WriteField(FieldDescriptor field, object? raw, WriteScope scope) =>
        field.WireWriter is not null ? field.WireWriter(raw) : WriteValue(raw, scope);

    private static object? WriteValue(object? value, WriteScope scope)
    {
        if (value is null)
        {
            return null;
        }

        Type type = value.GetType();

        if (scope.NativeTypes.Contains(type) || _passThroughTypes.Contains(type))
        {
            return value;
        }

        if (ConverterRegistry.TryResolve(type, out IWireConverter? converter))
        {
            return converter.ToWire(value);
        }

        if (value is Enum enumValue)
        {
            // After the registry lookup and on a branch of its own, exactly where Python's
            // `isinstance(value, Enum)` arm sits: an enum carries no single Serialization Name, so
            // it cannot be a registered converter (see EnumConverter). Routing through it is what
            // makes an [EnumValue]-annotated member write its declared string rather than its
            // ordinal — the difference between writing "green" and writing 1.
            return EnumConverter.ToWire(enumValue);
        }

        if (VersionableRegistry.TryGetByType(type, out VersionableMetadata? metadata))
        {
            return WriteVersionable(value, metadata, scope);
        }

        if (value is IDictionary map)
        {
            return WriteMap(map, scope);
        }

        if (value is IEnumerable sequence)
        {
            return WriteSequence(sequence, IsSet(type), scope);
        }

        if (value is ITuple tuple)
        {
            // A tuple is not IEnumerable, so it would otherwise fall off the end of this dispatch
            // and be unwritable — which is what it was until a save of a tuple-bearing type failed
            // on every text backend. It lowers to a list, as Python's tuple does.
            return WriteTuple(tuple, scope);
        }

        throw new UnsupportedTypeException(
            $"Cannot serialize '{type}' at field {Describe(scope.Path)}. Register an "
                + $"{nameof(IWireConverter)} for it, or declare the type [Versionable].");
    }

    private static Dictionary<string, object?> WriteVersionable(
        object value,
        VersionableMetadata metadata,
        WriteScope scope)
    {
        if (!scope.Visited.Add(value))
        {
            throw new CircularReferenceException(Describe(scope.Path), value.GetType());
        }

        string parentPath = scope.Path;
        try
        {
            Dictionary<string, object?> wire = new(metadata.Fields.Count + 1, StringComparer.Ordinal)
            {
                [VersionableEnvelope.WrappedKey] = EnvelopeCodec.Wrap(
                    new EnvelopeMetadata(metadata.Name, metadata.Version, metadata.Hash)),
            };

            // SkipDefaults is deliberately not applied here: Python applies it in `save()` to the
            // root object only, and a nested object that silently dropped fields would be
            // indistinguishable from one whose fields were never written.
            foreach (FieldDescriptor field in metadata.Fields)
            {
                scope.Path = parentPath.Length == 0 ? field.WireName : $"{parentPath}.{field.WireName}";
                wire[field.WireName] = WriteField(field, field.Getter(value), scope);
            }

            return wire;
        }
        finally
        {
            scope.Path = parentPath;
            scope.Visited.Remove(value);
        }
    }

    private static Dictionary<string, object?> WriteMap(IDictionary map, WriteScope scope)
    {
        Dictionary<string, object?> wire = new(map.Count, StringComparer.Ordinal);
        string parentPath = scope.Path;
        foreach (DictionaryEntry entry in map)
        {
            string key = KeyToWireName(entry.Key, parentPath);
            scope.Path = $"{parentPath}[{key}]";
            wire[key] = WriteValue(entry.Value, scope);
        }

        scope.Path = parentPath;
        return wire;
    }

    private static List<object?> WriteSequence(IEnumerable sequence, bool isSet, WriteScope scope)
    {
        List<object?> items = [.. sequence.Cast<object?>()];
        if (isSet)
        {
            // Sets have no wire order, so pick one: Python sorts by repr before writing
            // (`_serializeCollection`), and an unordered write would make every save of the same
            // object a different file.
            items.Sort(static (left, right) => string.CompareOrdinal(SortKey(left), SortKey(right)));
        }

        string parentPath = scope.Path;
        List<object?> wire = new(items.Count);
        for (int index = 0; index < items.Count; index++)
        {
            scope.Path = $"{parentPath}[{index}]";
            wire.Add(WriteValue(items[index], scope));
        }

        scope.Path = parentPath;
        return wire;
    }

    /// <summary>
    /// Lowers a tuple element-wise, the way Python lowers <c>tuple</c>: to a list.
    /// </summary>
    /// <remarks>
    /// <see cref="ITuple"/> rather than reflection over <c>ItemN</c> fields: it is the interface
    /// every <c>ValueTuple</c> and <c>Tuple</c> already implements, its indexer is a plain virtual
    /// call, and it flattens the nested <c>Rest</c> field of an eight-or-more element tuple by
    /// itself — so this stays reflection-free and AOT-safe (ADR-0003).
    /// <para>
    /// Only the write path needs this. Reading a tuple means constructing one, which is the
    /// generated <see cref="FieldDescriptor.WireReader"/>'s job; there is no
    /// <see cref="FieldDescriptor.WireWriter"/> for a tuple, because there does not need to be.
    /// </para>
    /// </remarks>
    private static List<object?> WriteTuple(ITuple tuple, WriteScope scope)
    {
        string parentPath = scope.Path;
        List<object?> wire = new(tuple.Length);
        for (int index = 0; index < tuple.Length; index++)
        {
            scope.Path = $"{parentPath}[{index}]";
            wire.Add(WriteValue(tuple[index], scope));
        }

        scope.Path = parentPath;
        return wire;
    }

    private static string SortKey(object? value) =>
        value is null ? string.Empty : Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty;

    private static string KeyToWireName(object? key, string path)
    {
        if (key is string text)
        {
            return text;
        }

        // Python stringifies dict keys with str(k); the declared key type is what turns them back
        // on load. A Versionable key would stringify to something no reader can invert, which
        // Python rejects outright — so does this.
        if (key is not null && VersionableRegistry.TryGetByType(key.GetType(), out _))
        {
            throw new ConverterException(
                $"Dictionary keys cannot be [Versionable] types ({key.GetType()}) at field "
                    + $"{Describe(path)}. Use them as values, not keys.");
        }

        return Convert.ToString(key, CultureInfo.InvariantCulture)
            ?? throw new ConverterException($"Dictionary key at field {Describe(path)} has no string form.");
    }

    // The annotation is what lets the trimmer keep the interface list of a type that only ever
    // arrives here as `value.GetType()`; without it IL2070 fires and, worse, a trimmed build could
    // silently decide a HashSet<T> is an ordinary sequence and stop sorting it.
    private static bool IsSet([DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.Interfaces)] Type type)
    {
        foreach (Type contract in type.GetInterfaces())
        {
            if (contract.IsGenericType)
            {
                Type definition = contract.GetGenericTypeDefinition();
                if (definition == typeof(ISet<>) || definition == typeof(IReadOnlySet<>))
                {
                    return true;
                }
            }
        }

        return false;
    }

    // ------------------------------------------------------------------
    // Read
    // ------------------------------------------------------------------

    /// <summary>
    /// Raises a wire value to <paramref name="declaredType"/>.
    /// </summary>
    /// <remarks>
    /// Handles what the engine can do without constructing a type it cannot see: primitives,
    /// registered converters, numeric enums, and nested <c>[Versionable]</c> objects (whose
    /// generated <see cref="VersionableMetadata.Factory"/> does the constructing). Containers,
    /// arrays, and value tuples throw here by design — building a <c>List&lt;T&gt;</c> at runtime
    /// means <c>Activator.CreateInstance</c> over a type the trimmer cannot see, which is why the
    /// generator emits a <see cref="FieldDescriptor.WireReader"/> for those fields instead
    /// (ADR-0003). Those readers call back into <see cref="ReadUnion"/>, <see cref="AsList"/>,
    /// <see cref="AsMap"/>, and this method for their elements.
    /// </remarks>
    /// <param name="wire">The value as the backend read it.</param>
    /// <param name="declaredType">
    /// The declared CLR type of the field. Annotated for the trimmer because an enum target has
    /// its members read reflectively by <see cref="EnumConverter"/>; a <c>typeof(T)</c> literal —
    /// which is what generated readers pass — satisfies it for free.
    /// </param>
    /// <returns>The materialized CLR value.</returns>
    /// <exception cref="ConverterException">The wire value is not valid for the declared type.</exception>
    /// <exception cref="UnsupportedTypeException">The declared type needs a generated reader.</exception>
    public static object? Read(
        object? wire,
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields)] Type declaredType)
    {
        ArgumentNullException.ThrowIfNull(declaredType);

        if (_readScope is not null)
        {
            return ReadValue(wire, declaredType, _readScope);
        }

        ReadScope scope = new();
        _readScope = scope;
        try
        {
            return ReadValue(wire, declaredType, scope);
        }
        finally
        {
            _readScope = null;
        }
    }

    /// <summary>
    /// Materializes a nested <c>[Versionable]</c> object from its wire dictionary.
    /// </summary>
    /// <remarks>
    /// Reads the object's own envelope, so a value declared as a base type comes back as the
    /// concrete type the file names — the polymorphic-collection case. Python counterpart:
    /// <c>_deserializeVersionable</c>.
    /// </remarks>
    /// <param name="wire">The nested wire dictionary, envelope included.</param>
    /// <param name="declaredMetadata">
    /// Metadata of the declared field type. Doubles as the upper bound for polymorphic
    /// resolution: the file's type must be assignable to it.
    /// </param>
    /// <returns>The materialized instance.</returns>
    public static object ReadVersionable(object? wire, VersionableMetadata declaredMetadata)
    {
        ArgumentNullException.ThrowIfNull(declaredMetadata);

        if (_readScope is not null)
        {
            return ObjectMaterializer.ReadNested(wire, declaredMetadata, _readScope);
        }

        ReadScope scope = new();
        _readScope = scope;
        try
        {
            return ObjectMaterializer.ReadNested(wire, declaredMetadata, scope);
        }
        finally
        {
            _readScope = null;
        }
    }

    /// <summary>
    /// Tries each member of a union in order and returns the first that materializes.
    /// </summary>
    /// <remarks>
    /// Python counterpart: <c>_deserializeUnion</c>. Multi-member unions (<c>int | str</c>) have
    /// no CLR type of their own, so the generator emits a call to this with one candidate per
    /// member; plain <c>T?</c> needs none, because <see langword="null"/> short-circuits and the
    /// declared type carries the rest.
    /// <para>
    /// A <see cref="DtypeMismatchException"/> is rethrown rather than treated as "wrong member".
    /// An array whose element type contradicts its declaration is schema drift, and swallowing it
    /// would silently fall through to a member that happens to accept the value — turning a
    /// declared-dtype violation into a quietly different field type.
    /// </para>
    /// </remarks>
    /// <param name="wire">The value as the backend read it.</param>
    /// <param name="candidates">One reader per union member, in canonical order.</param>
    /// <returns>
    /// The first successful result, or <paramref name="wire"/> unchanged when no member accepted
    /// it — which is what Python returns.
    /// </returns>
    public static object? ReadUnion(object? wire, params Func<object?, object?>[] candidates)
    {
        ArgumentNullException.ThrowIfNull(candidates);

        if (wire is null)
        {
            return null;
        }

        if (candidates.Length == 1)
        {
            // Python short-circuits a union with one non-None member straight into deserialize,
            // outside the try/except: with nothing to fall through to, swallowing the failure
            // would turn "this value is not a str" into "here is the raw value", and the wrong
            // type would surface later as a cast failure with no clue where it came from.
            return candidates[0](wire);
        }

        foreach (Func<object?, object?> candidate in candidates)
        {
            try
            {
                return candidate(wire);
            }
            catch (DtypeMismatchException)
            {
                throw;
            }
            catch (Exception error) when (error is ConverterException
                or InvalidCastException
                or FormatException
                or OverflowException
                or ArgumentException)
            {
                // Wrong member for this value — try the next one, as Python does.
            }
        }

        return wire;
    }

    /// <summary>Views a wire value as a list, for generated container readers.</summary>
    /// <param name="wire">A wire value written from a list, set, or tuple.</param>
    /// <returns>The elements, in file order.</returns>
    /// <exception cref="ConverterException">The wire value is not a sequence.</exception>
    public static IReadOnlyList<object?> AsList(object? wire) =>
        wire switch
        {
            IReadOnlyList<object?> list => list,
            string text => throw new ConverterException($"Expected a sequence on the wire, found the string '{text}'."),
            IEnumerable sequence => [.. sequence.Cast<object?>()],
            _ => throw new ConverterException(
                $"Expected a sequence on the wire, found {wire?.GetType().ToString() ?? "null"}."),
        };

    /// <summary>Views a wire value as a dictionary, for generated container readers.</summary>
    /// <param name="wire">A wire value written from a dictionary or a nested object.</param>
    /// <returns>The entries, keyed by wire name.</returns>
    /// <exception cref="ConverterException">The wire value is not a mapping.</exception>
    public static IReadOnlyDictionary<string, object?> AsMap(object? wire) =>
        wire switch
        {
            IReadOnlyDictionary<string, object?> map => map,
            IDictionary<string, object?> map => new Dictionary<string, object?>(map, StringComparer.Ordinal),
            _ => throw new ConverterException(
                $"Expected a mapping on the wire, found {wire?.GetType().ToString() ?? "null"}."),
        };

    internal static object? ReadValue(
        object? wire,
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields)] Type declaredType,
        ReadScope scope)
    {
        if (wire is null)
        {
            return null;
        }

        Type target = Nullable.GetUnderlyingType(declaredType) ?? declaredType;

        if (scope.NativeTypes.Contains(target) || target == typeof(object))
        {
            return wire;
        }

        if (_passThroughTypes.Contains(target))
        {
            return ReadPrimitive(wire, target);
        }

        if (ConverterRegistry.TryResolve(target, out IWireConverter? converter))
        {
            return converter.FromWire(wire, declaredType);
        }

        if (target.IsEnum)
        {
            // EnumConverter owns all three of Python's arms: the declared [EnumValue] string, the
            // bare number, and the [EnumFallback] member for a value this build no longer defines.
            return EnumConverter.FromWire(wire, target);
        }

        if (VersionableRegistry.TryGetByType(target, out VersionableMetadata? metadata))
        {
            return ObjectMaterializer.ReadNested(wire, metadata, scope);
        }

        throw new UnsupportedTypeException(
            $"Cannot materialize '{target}' from the wire. Containers, arrays, and tuples need the "
                + $"generated {nameof(FieldDescriptor.WireReader)}; other types need an "
                + $"{nameof(IWireConverter)}.");
    }

    internal static IDisposable EnterReadScope(ReadScope scope)
    {
        _readScope = scope;
        return new ReadScopeReleaser();
    }

    private static object ReadPrimitive(object wire, Type target)
    {
        if (target.IsInstanceOfType(wire))
        {
            return wire;
        }

        try
        {
            // Width erasure means every integral type is `int` on the wire, so a file written
            // from a `long` can hold a value this field cannot represent. Convert.ChangeType
            // checks the range; Python's int() would silently accept it and later truncate.
            return Convert.ChangeType(wire, target, CultureInfo.InvariantCulture);
        }
        catch (Exception error) when (error is OverflowException or FormatException or InvalidCastException)
        {
            throw new ConverterException(
                $"Cannot read {Quote(wire)} as '{target}': {error.Message}", error);
        }
    }

    internal static string Describe(string path) => path.Length == 0 ? "<root>" : path;

    internal static string Quote(object? value) =>
        value switch
        {
            null => "null",
            string text => $"'{text}'",
            _ => Convert.ToString(value, CultureInfo.InvariantCulture) ?? value.GetType().Name,
        };

    private sealed class ReadScopeReleaser : IDisposable
    {
        public void Dispose() => _readScope = null;
    }
}
