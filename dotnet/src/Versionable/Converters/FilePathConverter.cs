namespace Versionable.Converters;

/// <summary>
/// <see cref="FilePath"/> ⇄ its path string.
/// </summary>
/// <remarks>
/// Python counterpart: <c>registerConverter(Path, str, Path, matchSubclasses=True)</c> in
/// <c>src/versionable/_types.py</c>. Canonical name <c>Path</c> (GRAMMAR §9).
/// <para>
/// The string crosses verbatim in both directions — see <see cref="FilePath"/> for why no
/// separator normalisation happens here. Python's <c>PurePosixPath</c> and
/// <c>PureWindowsPath</c> converters have no C# counterpart in v1.
/// </para>
/// <para>
/// Python registers its <c>Path</c> converter with <c>matchSubclasses=True</c>, because
/// <c>pathlib.Path</c> instantiates as <c>PosixPath</c> or <c>WindowsPath</c> and the base
/// class never appears at run time. <see cref="FilePath"/> is sealed, so the exact match
/// already covers every value the field can hold and there is no subclass to route.
/// </para>
/// </remarks>
internal sealed class FilePathConverter : WireConverter<FilePath>
{
    /// <inheritdoc/>
    public override string SerializationName => "Path";

    /// <inheritdoc/>
    protected override object ToWireCore(FilePath value) => value.Value;

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType) =>
        new FilePath(RequireString(wireValue));
}
