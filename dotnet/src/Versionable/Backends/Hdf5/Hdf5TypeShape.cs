using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Numerics.Tensors;
using Versionable.Converters;

namespace Versionable.Backends.Hdf5;

/// <summary>What HDF5 construct a declared CLR type maps onto.</summary>
internal enum Hdf5ShapeKind
{
    /// <summary>Nothing is known about the field: infer from the value.</summary>
    Unknown = 0,

    /// <summary>An attribute — a primitive, an enum, or a converter's output.</summary>
    Scalar = 1,

    /// <summary>A dataset holding a real n-dimensional array.</summary>
    Tensor = 2,

    /// <summary>A 1-D dataset when the elements are scalars, otherwise an integer-keyed group.</summary>
    Sequence = 3,

    /// <summary>A group whose child names are the dictionary keys.</summary>
    Map = 4,

    /// <summary>A group carrying a <c>__versionable__</c> metadata child.</summary>
    Versionable = 5,
}

/// <summary>
/// The declared type of a field, reduced to the one fact the HDF5 layout depends on.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>fieldType</c> argument threaded through <c>_writeValue</c> and
/// <c>_readFields</c> in <c>src/versionable/_hdf5_backend.py</c>, plus the
/// <c>typing.get_origin</c> / <c>get_args</c> inspection those functions do inline. HDF5 is the
/// one backend whose <em>layout</em> — not just its values — depends on the annotation: a
/// <c>list[float]</c> is a dataset and a <c>list[Inner]</c> is a group, and nothing in the wire
/// value itself says which.
/// <para>
/// The order of the tests in <see cref="Of"/> is Python's order in <c>_writeValue</c> and is
/// load-bearing. In particular the converter lookup comes before the container tests, which is
/// what makes <c>byte[]</c> a base64 attribute rather than a dataset of small integers, and
/// <see cref="Complex"/> a two-element attribute rather than a dataset.
/// </para>
/// </remarks>
internal readonly struct Hdf5TypeShape
{
    private Hdf5TypeShape(Hdf5ShapeKind kind, Type? declaredType, Type? elementType)
    {
        Kind = kind;
        DeclaredType = declaredType;
        ElementType = elementType;
    }

    /// <summary>Nothing known; the writer and reader fall back to the value's own shape.</summary>
    internal static Hdf5TypeShape Unknown => new(Hdf5ShapeKind.Unknown, null, null);

    /// <summary>Which HDF5 construct the field maps onto.</summary>
    internal Hdf5ShapeKind Kind { get; }

    /// <summary>The declared type this shape was derived from, nullable wrapper removed.</summary>
    internal Type? DeclaredType { get; }

    /// <summary>
    /// Element type for <see cref="Hdf5ShapeKind.Sequence"/> and
    /// <see cref="Hdf5ShapeKind.Tensor"/>, value type for <see cref="Hdf5ShapeKind.Map"/>,
    /// otherwise <see langword="null"/>.
    /// </summary>
    internal Type? ElementType { get; }

    /// <summary>The shape of this shape's element type.</summary>
    internal Hdf5TypeShape Element => Of(ElementType);

    /// <summary>Reduces a declared field type to its HDF5 shape.</summary>
    /// <param name="declared">The declared type, or <see langword="null"/> when unknown.</param>
    /// <returns>The shape.</returns>
    internal static Hdf5TypeShape Of(Type? declared)
    {
        if (declared is null)
        {
            return Unknown;
        }

        Type type = Nullable.GetUnderlyingType(declared) ?? declared;

        if (TryTensorElement(type, out Type? tensorElement))
        {
            return new Hdf5TypeShape(Hdf5ShapeKind.Tensor, type, tensorElement);
        }

        if (VersionableRegistry.TryGetByType(type, out _))
        {
            return new Hdf5TypeShape(Hdf5ShapeKind.Versionable, type, null);
        }

        if (type == typeof(string) || type == typeof(object) || type.IsEnum || IsNumericScalar(type))
        {
            return new Hdf5TypeShape(Hdf5ShapeKind.Scalar, type, null);
        }

        // Before the container tests, as Python's `_registry.get(type(value))` arm sits before
        // its `isinstance(value, (list, set, ...))` arm.
        if (ConverterRegistry.TryResolve(type, out _))
        {
            return new Hdf5TypeShape(Hdf5ShapeKind.Scalar, type, null);
        }

        if (TryMapValueType(type, out Type? valueType))
        {
            return new Hdf5TypeShape(Hdf5ShapeKind.Map, type, valueType);
        }

        if (TryElementType(type, out Type? element))
        {
            return new Hdf5TypeShape(Hdf5ShapeKind.Sequence, type, element);
        }

        if (IsValueTuple(type))
        {
            // Python reads a tuple's element type from `get_args(...)[0]`, which for the
            // heterogeneous case is only the first member's type. Matching that is what keeps
            // `tuple[int, int]` a single int64 dataset on both sides.
            return new Hdf5TypeShape(Hdf5ShapeKind.Sequence, type, type.GetGenericArguments()[0]);
        }

        return new Hdf5TypeShape(Hdf5ShapeKind.Unknown, type, null);
    }

    /// <summary>
    /// Whether a sequence of <paramref name="elementType"/> is written as a 1-D dataset.
    /// </summary>
    /// <param name="elementType">The declared element type, or <see langword="null"/>.</param>
    /// <returns>
    /// <see langword="true"/> for the element types HDF5 stores as dataset elements. Python
    /// counterpart: <c>_isScalarType</c>, whose set is <c>{int, float, str, bool}</c> — the
    /// wider C# set is the same four with the integer and float widths spelled out.
    /// </returns>
    internal static bool IsDatasetElement([NotNullWhen(true)] Type? elementType) =>
        elementType is not null
        && (elementType == typeof(string) || IsNumericScalar(elementType));

    /// <summary>Whether <paramref name="type"/> is a closed <c>Tensor&lt;T&gt;</c>.</summary>
    /// <param name="type">The candidate type.</param>
    /// <param name="elementType">The tensor's element type.</param>
    /// <returns><see langword="true"/> when it is.</returns>
    internal static bool TryTensorElement(Type type, [NotNullWhen(true)] out Type? elementType)
    {
        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Tensor<>))
        {
            elementType = type.GetGenericArguments()[0];
            return true;
        }

        elementType = null;
        return false;
    }

    private static bool IsNumericScalar(Type type) =>
        type == typeof(bool)
        || type == typeof(sbyte) || type == typeof(byte)
        || type == typeof(short) || type == typeof(ushort)
        || type == typeof(int) || type == typeof(uint)
        || type == typeof(long) || type == typeof(ulong)
        || type == typeof(Half) || type == typeof(float) || type == typeof(double);

    private static bool IsValueTuple(Type type) =>
        type.IsGenericType
        && type.FullName is not null
        && type.FullName.StartsWith("System.ValueTuple`", StringComparison.Ordinal);

    // GetInterfaces() on a Type reached from a typeof() literal in generated metadata: no
    // members are read off the result, only its generic arguments, so nothing here needs a
    // DynamicallyAccessedMembers annotation.
    private static bool TryMapValueType(Type type, [NotNullWhen(true)] out Type? valueType)
    {
        foreach (Type candidate in Closures(type))
        {
            Type definition = candidate.GetGenericTypeDefinition();
            if (definition == typeof(IDictionary<,>) || definition == typeof(IReadOnlyDictionary<,>))
            {
                valueType = candidate.GetGenericArguments()[1];
                return true;
            }
        }

        valueType = null;
        return false;
    }

    private static bool TryElementType(Type type, [NotNullWhen(true)] out Type? elementType)
    {
        if (type.IsArray)
        {
            elementType = type.GetElementType();
            return elementType is not null;
        }

        foreach (Type candidate in Closures(type))
        {
            if (candidate.GetGenericTypeDefinition() == typeof(IEnumerable<>))
            {
                elementType = candidate.GetGenericArguments()[0];
                return true;
            }
        }

        elementType = null;
        return false;
    }

    // The Type reaching here is always a typeof() literal from generated metadata, so the type
    // is statically referenced and its interface list is rooted by that reference. The
    // annotation cannot be carried on the parameter instead: FieldDescriptor.ClrType — the sole
    // source — declares PublicFields (for the enum path), and requiring Interfaces here would
    // make every call site a warning that no contract change short of widening that member
    // could silence. WireValues.IsSet makes the same call on the write side.
    [UnconditionalSuppressMessage(
        "Trimming",
        "IL2070:UnrecognizedReflectionPattern",
        Justification = "Interface list of a typeof()-referenced type; only generic arguments are read.")]
    private static IEnumerable<Type> Closures(Type type)
    {
        if (type.IsGenericType)
        {
            yield return type;
        }

        foreach (Type contract in type.GetInterfaces())
        {
            if (contract.IsGenericType)
            {
                yield return contract;
            }
        }
    }
}
