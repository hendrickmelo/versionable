namespace Versionable.Errors;

/// <summary>
/// A migration needs to rewrite the file and the caller did not permit it.
/// </summary>
/// <remarks>
/// Python counterpart: <c>UpgradeRequiredError</c> in <c>src/versionable/errors.py</c>,
/// raised when a migration declares <c>requiresUpgrade()</c> but the load did not pass
/// <c>upgradeInPlace=True</c>.
/// </remarks>
public class UpgradeRequiredException : VersionableException
{
    /// <summary>Initializes a new instance of the <see cref="UpgradeRequiredException"/> class.</summary>
    public UpgradeRequiredException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="UpgradeRequiredException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    public UpgradeRequiredException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="UpgradeRequiredException"/> class.</summary>
    /// <param name="message">Message describing the failure.</param>
    /// <param name="innerException">The underlying failure.</param>
    public UpgradeRequiredException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
