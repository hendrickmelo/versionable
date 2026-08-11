namespace Versionable.Tests;

/// <summary>A scratch <c>.h5</c> path that deletes itself.</summary>
/// <remarks>
/// PureHDF writes through a real <see cref="FileStream"/>, so the HDF5 suites cannot work in
/// memory the way the text-backend ones can.
/// </remarks>
internal sealed class Hdf5TempFile : IDisposable
{
    /// <summary>Initializes a new instance of the <see cref="Hdf5TempFile"/> class.</summary>
    /// <param name="extension">File extension including the dot.</param>
    internal Hdf5TempFile(string extension = ".h5") =>
        Path = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            $"versionable-{Guid.NewGuid():N}{extension}");

    /// <summary>The path. Nothing exists there until something writes it.</summary>
    internal string Path { get; }

    /// <summary>Size of the written file in bytes.</summary>
    internal long Length => new FileInfo(Path).Length;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (File.Exists(Path))
        {
            File.Delete(Path);
        }
    }
}
