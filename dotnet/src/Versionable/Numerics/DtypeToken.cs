namespace Versionable.Numerics;

/// <summary>
/// The closed set of array element types the wire format admits.
/// </summary>
/// <remarks>
/// Exactly the rows of the dtype table in <c>conformance/GRAMMAR.md</c> §7, in that order —
/// the enum values are used as indices into the safe-cast table in <see cref="Dtypes"/>, so
/// the order is load-bearing and must not be reshuffled.
/// <para>
/// Python counterpart: the <c>_DTYPE_TOKENS</c> table in <c>src/versionable/_arrays.py</c>.
/// The set is closed on purpose: a numpy release cannot silently add a dtype that hashes to a
/// token no other implementation knows.
/// </para>
/// </remarks>
internal enum DtypeToken
{
    /// <summary><c>bool</c> — one byte per element, <c>0</c> or <c>1</c>.</summary>
    Bool = 0,

    /// <summary><c>int8</c> — <see cref="sbyte"/>.</summary>
    Int8 = 1,

    /// <summary><c>int16</c> — <see cref="short"/>.</summary>
    Int16 = 2,

    /// <summary><c>int32</c> — <see cref="int"/>.</summary>
    Int32 = 3,

    /// <summary><c>int64</c> — <see cref="long"/>.</summary>
    Int64 = 4,

    /// <summary><c>uint8</c> — <see cref="byte"/>.</summary>
    UInt8 = 5,

    /// <summary><c>uint16</c> — <see cref="ushort"/>.</summary>
    UInt16 = 6,

    /// <summary><c>uint32</c> — <see cref="uint"/>.</summary>
    UInt32 = 7,

    /// <summary><c>uint64</c> — <see cref="ulong"/>.</summary>
    UInt64 = 8,

    /// <summary><c>float16</c> — <see cref="Half"/>.</summary>
    Float16 = 9,

    /// <summary><c>float32</c> — <see cref="float"/>.</summary>
    Float32 = 10,

    /// <summary><c>float64</c> — <see cref="double"/>.</summary>
    Float64 = 11,

    /// <summary>
    /// <c>complex64</c> — a pair of <c>float32</c>. Python-only as a declared type: .NET has
    /// no single-precision complex, so it is decoded into
    /// <see cref="System.Numerics.Complex"/> (a safe widening cast) and never written by C#.
    /// </summary>
    Complex64 = 12,

    /// <summary><c>complex128</c> — <see cref="System.Numerics.Complex"/>.</summary>
    Complex128 = 13,
}
