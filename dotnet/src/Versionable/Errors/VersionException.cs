namespace Versionable.Errors;

/// <summary>
/// A file's schema version cannot be reconciled with the target type's version.
/// </summary>
/// <remarks>
/// Python counterpart: <c>VersionError</c> in <c>src/versionable/errors.py</c>. Typically a
/// version gap with no migration to bridge it, or a file newer than the code reading it.
/// </remarks>
public class VersionException : VersionableException
{
    /// <summary>Initializes a new instance of the <see cref="VersionException"/> class.</summary>
    public VersionException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="VersionException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    public VersionException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="VersionException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    /// <param name="innerException">The underlying failure.</param>
    public VersionException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
