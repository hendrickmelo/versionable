namespace Versionable.Errors;

/// <summary>
/// A migration could not be applied.
/// </summary>
/// <remarks>
/// Python counterpart: <c>MigrationError</c> in <c>src/versionable/errors.py</c>.
/// </remarks>
public class MigrationException : VersionableException
{
    /// <summary>Initializes a new instance of the <see cref="MigrationException"/> class.</summary>
    public MigrationException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="MigrationException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    public MigrationException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="MigrationException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    /// <param name="innerException">The underlying failure.</param>
    public MigrationException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
