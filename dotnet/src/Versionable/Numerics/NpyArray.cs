namespace Versionable.Numerics;

/// <summary>
/// A decoded NPY payload: a dtype, a C-order shape, and the flat element buffer.
/// </summary>
/// <remarks>
/// <see cref="Values"/> is always one-dimensional and always in C (row-major) order,
/// regardless of <see cref="Shape"/>: the shape is carried alongside rather than baked into a
/// multidimensional CLR array because <c>Tensor&lt;T&gt;</c> is built from a flat buffer plus
/// lengths, and reshaping in between would be pure copying.
/// </remarks>
internal sealed class NpyArray
{
    /// <summary>Initializes a new instance of the <see cref="NpyArray"/> class.</summary>
    /// <param name="dtype">The element dtype.</param>
    /// <param name="shape">The C-order shape; empty for a 0-d (scalar) array.</param>
    /// <param name="values">The flat element buffer, in C order.</param>
    internal NpyArray(DtypeToken dtype, long[] shape, Array values)
    {
        Dtype = dtype;
        Shape = shape;
        Values = values;
    }

    /// <summary>The element dtype.</summary>
    internal DtypeToken Dtype { get; }

    /// <summary>The C-order shape. Empty for a 0-d array, which still holds one element.</summary>
    internal long[] Shape { get; }

    /// <summary>The flat element buffer, in C order, typed per <see cref="Dtypes.ToClrType"/>.</summary>
    internal Array Values { get; }
}
