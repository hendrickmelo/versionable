namespace Versionable.Errors;

/// <summary>
/// Base of every exception the Versionable runtime raises.
/// </summary>
/// <remarks>
/// Python counterpart: <c>VersionableError</c> in <c>src/versionable/errors.py</c>. The C#
/// hierarchy mirrors that tree one-for-one, with the <c>Error</c> suffix renamed to
/// <c>Exception</c> per .NET convention. Also raised directly for registry collisions, the
/// analogue of Python's bare <c>VersionableError</c> in <c>__init_subclass__</c>.
/// </remarks>
public class VersionableException : Exception
{
    /// <summary>Initializes a new instance of the <see cref="VersionableException"/> class.</summary>
    public VersionableException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="VersionableException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    public VersionableException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="VersionableException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    /// <param name="innerException">The underlying failure.</param>
    public VersionableException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
