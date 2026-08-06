namespace Versionable.Backends.Hdf5;

/// <summary>
/// Turns the two compression-filter failures into messages that name the cause.
/// </summary>
/// <remarks>
/// Python counterpart: <c>missingFilterHint</c> in <c>src/versionable/_hdf5_plugin.py</c>,
/// which appends the same kind of note to a read failure caused by a filter <c>hdf5plugin</c>
/// has not registered. Applied on <em>both</em> sides here, because a C# process can hit it
/// writing (the native library is missing) as easily as reading (the file's filter is one
/// nothing registered), and the raw failure is unreadable either way: a
/// <see cref="DllNotFoundException"/> nested three deep inside a reflection invocation, or an
/// HDF5 filter id with no name attached.
/// </remarks>
internal static class Hdf5Diagnostics
{
    /// <summary>Returns a hint to append to a failure message, or the empty string.</summary>
    /// <param name="error">The failure, whose whole inner chain is inspected.</param>
    /// <returns>A sentence starting with a space, or the empty string.</returns>
    internal static string FilterHint(Exception error)
    {
        for (Exception? cause = error; cause is not null; cause = cause.InnerException)
        {
            if (cause is DllNotFoundException)
            {
                return " The compression filter needs a native library that is not available on "
                    + "this platform (Blosc2.PInvoke ships binaries for win-x86, win-x64, and "
                    + "linux-x64 only). Use Hdf5Compression.Gzip, which is built into HDF5 and "
                    + "needs nothing native.";
            }

            if (cause.Message.Contains("filter", StringComparison.OrdinalIgnoreCase))
            {
                return " This can mean the file uses a compression filter this build does not "
                    + "register. Gzip, shuffle, and Blosc v1 (filter 32001) are available; Blosc2 "
                    + "(32026), Zstd (32015), and LZF (32000) — which Python can write through "
                    + "hdf5plugin — are not. Re-save the file with gzip from Python to read it "
                    + "here.";
            }
        }

        return string.Empty;
    }
}
