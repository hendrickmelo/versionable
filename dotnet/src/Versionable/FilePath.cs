namespace Versionable;

/// <summary>
/// A filesystem path on the wire: the C# counterpart of Python's <c>pathlib.Path</c>.
/// </summary>
/// <remarks>
/// Canonical name <c>Path</c> (<c>conformance/GRAMMAR.md</c> §9); the wire form is the path
/// string, verbatim.
/// <para>
/// It exists because .NET has no path type — paths are bare <see cref="string"/>s — and a
/// bare string canonicalises to <c>str</c>, which would hash differently from the Python
/// schema it mirrors. Declaring a field as <see cref="FilePath"/> is what makes it hash
/// <c>Path</c>.
/// </para>
/// <para>
/// The string is stored and written <em>verbatim</em>: no separator normalisation, no
/// rooting, no case folding. Python has the same property in the other direction — a
/// <c>Path</c> round-trips through <c>str(v)</c>, so a file written on POSIX carries forward
/// slashes and one written on Windows carries backslashes, and neither side rewrites the
/// other's separators. Normalising here would make the wire value depend on the writing
/// machine's OS, which is exactly what a byte-comparable file format cannot afford. Callers
/// that need a native path should pass <see cref="Value"/> through
/// <see cref="System.IO.Path"/>.
/// </para>
/// <para>
/// Python's <c>PurePosixPath</c> and <c>PureWindowsPath</c> have no C# counterpart in v1
/// (GRAMMAR §9). A file whose schema uses them cannot be mirrored in C# without renaming the
/// field's type on the Python side.
/// </para>
/// </remarks>
[SerializationName("Path")]
public sealed class FilePath : IEquatable<FilePath>
{
    /// <summary>Initializes a new instance of the <see cref="FilePath"/> class.</summary>
    /// <param name="value">The path string, stored verbatim.</param>
    public FilePath(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        Value = value;
    }

    /// <summary>The path string, exactly as supplied and exactly as written to the file.</summary>
    public string Value { get; }

    /// <summary>Wraps <paramref name="value"/> in a <see cref="FilePath"/>.</summary>
    /// <param name="value">The path string.</param>
    public static implicit operator FilePath(string value) => new(value);

    /// <summary>Unwraps the path string.</summary>
    /// <param name="path">The path to unwrap.</param>
    public static explicit operator string(FilePath path)
    {
        ArgumentNullException.ThrowIfNull(path);
        return path.Value;
    }

    /// <summary>Compares two paths by their string value, ordinally.</summary>
    /// <param name="left">Left operand.</param>
    /// <param name="right">Right operand.</param>
    /// <returns><see langword="true"/> when both are null or hold the same string.</returns>
    public static bool operator ==(FilePath? left, FilePath? right) =>
        left is null ? right is null : left.Equals(right);

    /// <summary>Negation of <see cref="op_Equality"/>.</summary>
    /// <param name="left">Left operand.</param>
    /// <param name="right">Right operand.</param>
    /// <returns><see langword="true"/> when the two differ.</returns>
    public static bool operator !=(FilePath? left, FilePath? right) => !(left == right);

    /// <summary>Named alternative to the implicit string conversion.</summary>
    /// <param name="value">The path string.</param>
    /// <returns>A <see cref="FilePath"/> wrapping <paramref name="value"/>.</returns>
    public static FilePath FromString(string value) => new(value);

    /// <inheritdoc/>
    public bool Equals(FilePath? other) =>
        other is not null && string.Equals(Value, other.Value, StringComparison.Ordinal);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as FilePath);

    /// <inheritdoc/>
    public override int GetHashCode() => StringComparer.Ordinal.GetHashCode(Value);

    /// <summary>Returns the path string.</summary>
    /// <returns><see cref="Value"/>.</returns>
    public override string ToString() => Value;
}
