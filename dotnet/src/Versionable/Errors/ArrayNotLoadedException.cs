namespace Versionable.Errors;

/// <summary>
/// An array field was read that was deliberately not materialized at load time.
/// </summary>
/// <remarks>
/// Python counterpart: <c>ArrayNotLoadedError</c> in <c>src/versionable/errors.py</c>,
/// which also inherits <c>AttributeError</c> so <c>hasattr()</c> reports
/// <see langword="false"/>. C# has no equivalent probing protocol, so this derives from
/// <see cref="VersionableException"/> only.
/// </remarks>
public class ArrayNotLoadedException : VersionableException
{
    /// <summary>Initializes a new instance of the <see cref="ArrayNotLoadedException"/> class.</summary>
    public ArrayNotLoadedException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="ArrayNotLoadedException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    public ArrayNotLoadedException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="ArrayNotLoadedException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    /// <param name="innerException">The underlying failure.</param>
    public ArrayNotLoadedException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
