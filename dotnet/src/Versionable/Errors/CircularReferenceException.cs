namespace Versionable.Errors;

/// <summary>
/// The object graph reaches itself: an instance already on the serialization stack was
/// encountered again.
/// </summary>
/// <remarks>
/// Python counterpart: <c>CircularReferenceError</c> in <c>src/versionable/errors.py</c>,
/// raised from <c>_serializeVersionable</c> in <c>src/versionable/_types.py</c>. Diamonds
/// (the same instance reached from two unrelated branches) are not cycles and are permitted;
/// they are duplicated on disk.
/// </remarks>
public class CircularReferenceException : VersionableException
{
    /// <summary>Initializes a new instance of the <see cref="CircularReferenceException"/> class.</summary>
    public CircularReferenceException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="CircularReferenceException"/> class.</summary>
    /// <param name="message">Message describing the cycle.</param>
    public CircularReferenceException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="CircularReferenceException"/> class.</summary>
    /// <param name="message">Message describing the cycle.</param>
    /// <param name="innerException">The underlying failure.</param>
    public CircularReferenceException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="CircularReferenceException"/> class.</summary>
    /// <param name="fieldPath">Field path to the revisited instance, e.g. <c>children[0]</c>. Empty for the root.</param>
    /// <param name="objectType">Type of the revisited instance.</param>
    public CircularReferenceException(string fieldPath, Type objectType)
        : base($"Circular reference detected at field path {(fieldPath.Length == 0 ? "<root>" : fieldPath)} "
            + $"→ {objectType?.Name}. Versionable cannot serialize cycles.")
    {
        FieldPath = fieldPath;
        ObjectType = objectType;
    }

    /// <summary>Field path to the revisited instance; empty for the root.</summary>
    public string? FieldPath { get; }

    /// <summary>Type of the revisited instance.</summary>
    public Type? ObjectType { get; }
}
