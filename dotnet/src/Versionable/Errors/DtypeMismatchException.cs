namespace Versionable.Errors;

/// <summary>
/// Which side of a round trip a dtype check ran on. Used only in messages.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>DtypeContext</c> literal alias in
/// <c>src/versionable/errors.py</c>.
/// </remarks>
public enum DtypeContext
{
    /// <summary>The array was being written.</summary>
    Save = 0,

    /// <summary>The array was being read.</summary>
    Load = 1,
}

/// <summary>
/// An array's element type cannot be safely cast to the dtype its declaration names.
/// </summary>
/// <remarks>
/// Python counterpart: <c>DtypeMismatchError</c> in <c>src/versionable/errors.py</c>. Array
/// dtype is hash-significant (ADR-0002): <c>Tensor&lt;double&gt;</c> canonicalises to
/// <c>ndarray[float64]</c>. Safe widening casts are applied silently; anything lossy raises
/// this rather than letting the file disagree with the schema its hash describes.
/// </remarks>
public class DtypeMismatchException : ConverterException
{
    /// <summary>Initializes a new instance of the <see cref="DtypeMismatchException"/> class.</summary>
    public DtypeMismatchException()
    {
    }

    /// <summary>Initializes a new instance of the <see cref="DtypeMismatchException"/> class.</summary>
    /// <param name="message">Message describing the mismatch.</param>
    public DtypeMismatchException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="DtypeMismatchException"/> class.</summary>
    /// <param name="message">Message describing the mismatch.</param>
    /// <param name="innerException">The underlying failure.</param>
    public DtypeMismatchException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="DtypeMismatchException"/> class.</summary>
    /// <param name="declared">Canonical name of the declared dtype, e.g. <c>float32</c>.</param>
    /// <param name="actual">Canonical name of the value's dtype.</param>
    /// <param name="fieldPath">Dotted path of the offending field; empty for the root.</param>
    /// <param name="context">Whether the check ran on save or on load.</param>
    public DtypeMismatchException(string declared, string actual, string fieldPath, DtypeContext context)
        : base($"Array dtype mismatch at {(fieldPath.Length == 0 ? "<root>" : fieldPath)} during "
            + $"{(context == DtypeContext.Save ? "save" : "load")}: declared ndarray[{declared}], "
            + $"data is {actual}. Casting {actual} to {declared} is not safe. Cast explicitly if the "
            + $"loss is intended, or declare ndarray[{actual}] and update the schema hash.")
    {
        Declared = declared;
        Actual = actual;
        FieldPath = fieldPath;
        Context = context;
    }

    /// <summary>Canonical name of the declared dtype.</summary>
    public string? Declared { get; }

    /// <summary>Canonical name of the value's dtype.</summary>
    public string? Actual { get; }

    /// <summary>Dotted path of the offending field; empty for the root.</summary>
    public string? FieldPath { get; }

    /// <summary>Whether the check ran on save or on load.</summary>
    public DtypeContext Context { get; }
}
