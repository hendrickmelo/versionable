namespace Versionable.Errors;

/// <summary>
/// A declared type cannot be expressed in the canonical type grammar.
/// </summary>
/// <remarks>
/// Python counterpart: <c>UnsupportedTypeError</c> in <c>src/versionable/errors.py</c>.
/// Raised for constructs the language-neutral grammar deliberately closes off — a literal
/// option that is not a string, int, bool, <see langword="null"/> or enum member, and array
/// dtypes outside the supported token table. In C# most of these are caught at compile time
/// by the analyzer; this exception covers the cases only the runtime sees.
/// </remarks>
public class UnsupportedTypeException : VersionableException
{
    /// <summary>Initializes a new instance of the <see cref="UnsupportedTypeException"/> class.</summary>
    public UnsupportedTypeException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="UnsupportedTypeException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    public UnsupportedTypeException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="UnsupportedTypeException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    /// <param name="innerException">The underlying failure.</param>
    public UnsupportedTypeException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
