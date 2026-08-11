namespace Versionable.Errors;

/// <summary>
/// A type conversion failed while serializing or deserializing.
/// </summary>
/// <remarks>
/// Python counterpart: <c>ConverterError</c> in <c>src/versionable/errors.py</c>.
/// </remarks>
public class ConverterException : VersionableException
{
    /// <summary>Initializes a new instance of the <see cref="ConverterException"/> class.</summary>
    public ConverterException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="ConverterException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    public ConverterException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="ConverterException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    /// <param name="innerException">The underlying failure.</param>
    public ConverterException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
