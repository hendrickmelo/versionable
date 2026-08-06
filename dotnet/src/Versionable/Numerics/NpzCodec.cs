using System.IO.Compression;
using Versionable.Errors;

namespace Versionable.Numerics;

/// <summary>
/// Reads and writes the numpy <c>.npz</c> container — a ZIP of <c>.npy</c> entries.
/// </summary>
/// <remarks>
/// The wire form of an ndarray in the text backends is base64 of one of these
/// (<c>_serializeNdarray</c> in <c>src/versionable/_types.py</c>, which calls
/// <c>numpy.savez_compressed(buf, data=arr)</c> and reads back with
/// <c>numpy.load(buf)["data"]</c>). Two details of that call are the interchange contract and
/// are derived from it, not chosen here:
/// <list type="bullet">
/// <item>the single entry is named <c>data.npy</c> — <c>savez</c> appends <c>.npy</c> to the
/// keyword argument's name;</item>
/// <item>entries are DEFLATE-compressed — <c>savez_compressed</c> passes
/// <c>ZIP_DEFLATED</c>, where plain <c>savez</c> would store.</item>
/// </list>
/// <para>
/// <strong>Reads are exact; writes are equivalent, not byte-identical.</strong> The NPY
/// payload inside is byte-identical to numpy's, but the ZIP framing differs: CPython's
/// <c>zipfile</c> writes the local header with ZIP64 sentinel sizes
/// (<c>numpy.savez</c> passes <c>force_zip64=True</c>) and a Unix "version made by", while
/// <see cref="ZipArchive"/> writes plain 32-bit sizes. The DEFLATE bitstream may also differ:
/// both sides use zlib level 6, but not the same zlib. Both archives decompress to the same
/// NPY bytes, which is what interchange needs; the golden corpus is never regenerated from
/// the C# side.
/// </para>
/// </remarks>
internal static class NpzCodec
{
    /// <summary>
    /// The entry name Python writes and expects, without the <c>.npy</c> suffix.
    /// </summary>
    internal const string DefaultEntryName = "data";

    // zipfile's default when it creates a ZipInfo without a real file behind it. Matching it
    // keeps the timestamp out of the written bytes, so the same array encodes to the same
    // base64 on every run instead of embedding the wall clock.
    private static readonly DateTimeOffset _dosEpoch = new(1980, 1, 1, 0, 0, 0, TimeSpan.Zero);

    /// <summary>Decodes the single named array out of an NPZ container.</summary>
    /// <param name="npz">The complete <c>.npz</c> file.</param>
    /// <param name="entryName">The array name, without the <c>.npy</c> suffix.</param>
    /// <returns>The dtype, shape, and flat element buffer.</returns>
    /// <exception cref="ConverterException">The container is not a ZIP, or has no such entry.</exception>
    internal static NpyArray Decode(byte[] npz, string entryName = DefaultEntryName)
    {
        ArgumentNullException.ThrowIfNull(npz);

        string fileName = entryName + ".npy";
        try
        {
            using MemoryStream source = new(npz, writable: false);
            using ZipArchive archive = new(source, ZipArchiveMode.Read);
            ZipArchiveEntry? entry = archive.GetEntry(fileName);
            if (entry is null)
            {
                throw new ConverterException(
                    $"NPZ container has no '{fileName}' entry. Entries present: "
                    + $"{string.Join(", ", archive.Entries.Select(e => e.FullName))}.");
            }

            using Stream entryStream = entry.Open();

            // entry.Length is the central directory's claim, not a measurement, so it only
            // sizes the initial buffer and is capped: a doctored header should not be able to
            // ask for a gigabyte before a single byte has been decompressed.
            using MemoryStream buffer = new(capacity: (int)Math.Clamp(entry.Length, 0, 1 << 20));
            entryStream.CopyTo(buffer);
            return NpyCodec.Decode(buffer.ToArray());
        }
        catch (InvalidDataException e)
        {
            throw new ConverterException($"Not a readable NPZ container: {e.Message}", e);
        }
    }

    /// <summary>Encodes one array as a compressed NPZ container.</summary>
    /// <param name="dtype">The element dtype.</param>
    /// <param name="shape">The C-order shape.</param>
    /// <param name="values">The flat element buffer, in C order.</param>
    /// <param name="entryName">The array name, without the <c>.npy</c> suffix.</param>
    /// <returns>The complete <c>.npz</c> file.</returns>
    internal static byte[] Encode(
        DtypeToken dtype,
        long[] shape,
        Array values,
        string entryName = DefaultEntryName)
    {
        byte[] npy = NpyCodec.Encode(dtype, shape, values);
        using MemoryStream destination = new();
        using (ZipArchive archive = new(destination, ZipArchiveMode.Create, leaveOpen: true))
        {
            ZipArchiveEntry entry = archive.CreateEntry(entryName + ".npy", CompressionLevel.Optimal);
            entry.LastWriteTime = _dosEpoch;
            using Stream entryStream = entry.Open();
            entryStream.Write(npy, 0, npy.Length);
        }

        return destination.ToArray();
    }
}
