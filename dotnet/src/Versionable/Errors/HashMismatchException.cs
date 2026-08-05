namespace Versionable.Errors;

/// <summary>
/// A declared schema hash disagrees with the hash computed from the actual schema.
/// </summary>
/// <remarks>
/// Python counterpart: <c>HashMismatchError</c> in <c>src/versionable/errors.py</c>, raised
/// at class definition time. In C# the equivalent check is the analyzer's compile-time
/// diagnostic <c>VSN0001</c> (ADR-0003), so this exception is reserved for the runtime
/// cases the compiler cannot see — chiefly a file whose envelope hash does not match the
/// type reading it.
/// </remarks>
public class HashMismatchException : VersionableException
{
    /// <summary>Initializes a new instance of the <see cref="HashMismatchException"/> class.</summary>
    public HashMismatchException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="HashMismatchException"/> class.</summary>
    /// <param name="message">Message describing the mismatch.</param>
    public HashMismatchException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="HashMismatchException"/> class.</summary>
    /// <param name="message">Message describing the mismatch.</param>
    /// <param name="innerException">The underlying failure.</param>
    public HashMismatchException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="HashMismatchException"/> class.</summary>
    /// <param name="typeName">Serialization Name or CLR name of the type involved.</param>
    /// <param name="declared">The hash that was declared or read from the file.</param>
    /// <param name="computed">The hash computed from the current schema.</param>
    public HashMismatchException(string typeName, string declared, string computed)
        : base($"{typeName}: hash mismatch — declared '{declared}', computed '{computed}'. "
            + $"Update the declared hash to '{computed}'.")
    {
        TypeName = typeName;
        Declared = declared;
        Computed = computed;
    }

    /// <summary>Serialization Name or CLR name of the type involved.</summary>
    public string? TypeName { get; }

    /// <summary>The hash that was declared or read from the file.</summary>
    public string? Declared { get; }

    /// <summary>The hash computed from the current schema.</summary>
    public string? Computed { get; }
}
