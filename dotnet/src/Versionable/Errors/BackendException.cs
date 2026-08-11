namespace Versionable.Errors;

/// <summary>
/// A storage backend operation failed.
/// </summary>
/// <remarks>
/// Python counterpart: <c>BackendError</c> in <c>src/versionable/errors.py</c>. Also raised
/// when no backend is registered for a path's extension.
/// </remarks>
public class BackendException : VersionableException
{
    /// <summary>Initializes a new instance of the <see cref="BackendException"/> class.</summary>
    public BackendException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="BackendException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    public BackendException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="BackendException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    /// <param name="innerException">The underlying failure.</param>
    public BackendException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
