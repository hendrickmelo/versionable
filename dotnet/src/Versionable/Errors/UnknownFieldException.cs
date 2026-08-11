namespace Versionable.Errors;

/// <summary>
/// A file contains a field the target type does not declare.
/// </summary>
/// <remarks>
/// Python counterpart: <c>UnknownFieldError</c> in <c>src/versionable/errors.py</c>. Only
/// raised for types declared with <see cref="UnknownFieldPolicy.Error"/>.
/// </remarks>
public class UnknownFieldException : VersionableException
{
    /// <summary>Initializes a new instance of the <see cref="UnknownFieldException"/> class.</summary>
    public UnknownFieldException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="UnknownFieldException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    public UnknownFieldException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="UnknownFieldException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    /// <param name="innerException">The underlying failure.</param>
    public UnknownFieldException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
