using System.Globalization;
using PureHDF;
using Versionable.Engine;
using Versionable.Errors;

namespace Versionable.Backends.Hdf5;

/// <summary>
/// Builds the in-memory HDF5 tree for one object and writes it.
/// </summary>
/// <remarks>
/// Python counterpart: <c>_writeFields</c>, <c>_writeValue</c>, <c>_writeSequence</c>,
/// <c>_writeDict</c>, and <c>_writeVersionable</c> in
/// <c>src/versionable/_hdf5_backend.py</c>. The dispatch order in <see cref="WriteValue"/> is
/// that file's order, and the layout each arm produces is what the golden <c>.h5</c> files
/// hold.
/// <para>
/// The values reaching here have already been lowered by
/// <see cref="Engine.WireValues.WriteFields"/>, with every <c>Tensor&lt;T&gt;</c> declared
/// native and therefore passed through untouched. That is the division of labour the
/// <see cref="IVersionableBackend"/> contract asks for: the engine owns <em>what</em> a value
/// becomes, this file owns <em>where</em> in the file it goes — and the second question is the
/// one HDF5 answers differently from every text backend.
/// </para>
/// </remarks>
internal static class Hdf5Writer
{
    // Python (h5py) writes small datasets contiguous; PureHDF prefers the compact layout, which
    // parks the data inside the object header. Both are readable everywhere, but matching h5py
    // keeps a C#-written file and a Python-written one structurally identical, which is what
    // makes `h5diff` and `h5ls -v` usable as review tools across the two implementations.
    private static readonly H5WriteOptions _options = new() { PreferCompactDatasetLayout = false };

    /// <summary>Writes one object to <paramref name="path"/>.</summary>
    /// <param name="wire">Lowered field values keyed by wire name.</param>
    /// <param name="envelope">The envelope to record.</param>
    /// <param name="metadata">Metadata of the type being written; supplies declared field types.</param>
    /// <param name="path">Output file path.</param>
    /// <param name="compression">Filter settings for array datasets.</param>
    /// <exception cref="BackendException">The write failed.</exception>
    internal static void Write(
        IReadOnlyDictionary<string, object?> wire,
        EnvelopeMetadata envelope,
        VersionableMetadata metadata,
        string path,
        Hdf5Compression compression)
    {
        H5File file = [];
        WriteObject(file, wire, envelope, metadata, compression, string.Empty);

        try
        {
            file.Write(path, _options);
        }
        catch (Exception error) when (error is not VersionableException)
        {
            throw new BackendException(
                $"Failed to write HDF5 to '{path}': {error.Message}{Hdf5Diagnostics.FilterHint(error)}",
                error);
        }
    }

    private static void WriteObject(
        H5Group target,
        IReadOnlyDictionary<string, object?> wire,
        EnvelopeMetadata envelope,
        VersionableMetadata? metadata,
        Hdf5Compression compression,
        string path)
    {
        // The envelope lives in a child group rather than in attributes on the object itself, so
        // a Versionable group is distinguishable from a plain collection group by structure
        // alone — which is exactly what the reader dispatches on.
        target[VersionableEnvelope.WrappedKey] = MetadataGroup(envelope);

        HashSet<string> written = new(wire.Count, StringComparer.Ordinal);
        if (metadata is not null)
        {
            foreach (FieldDescriptor field in metadata.Fields)
            {
                if (wire.TryGetValue(field.WireName, out object? value))
                {
                    written.Add(field.WireName);
                    WriteValue(
                        target,
                        field.WireName,
                        value,
                        Hdf5TypeShape.Of(field.ClrType),
                        compression,
                        Join(path, field.WireName));
                }
            }
        }

        // Keys the type does not declare: an `unknown = Preserve` load put them there, and
        // dropping them would make preserve mean "preserve until the next save".
        foreach (KeyValuePair<string, object?> extra in wire)
        {
            if (!written.Contains(extra.Key) && extra.Key != VersionableEnvelope.WrappedKey)
            {
                WriteValue(
                    target, extra.Key, extra.Value, Hdf5TypeShape.Unknown, compression, Join(path, extra.Key));
            }
        }
    }

    private static H5Group MetadataGroup(EnvelopeMetadata envelope)
    {
        H5Group meta = [];
        if (envelope.ObjectName is not null)
        {
            meta.Attributes[VersionableEnvelope.ObjectKey] = Hdf5Values.ToAttribute(envelope.ObjectName, string.Empty);
        }

        if (envelope.Version is int version)
        {
            meta.Attributes[VersionableEnvelope.VersionKey] = Hdf5Values.ToAttribute((long)version, string.Empty);
        }

        if (envelope.Hash is not null)
        {
            meta.Attributes[VersionableEnvelope.HashKey] = Hdf5Values.ToAttribute(envelope.Hash, string.Empty);
        }

        return meta;
    }

    private static void WriteValue(
        H5Group parent,
        string name,
        object? value,
        Hdf5TypeShape shape,
        Hdf5Compression compression,
        string path)
    {
        if (value is null)
        {
            parent.Attributes[name] = Hdf5Values.ToAttribute(null, path);
            return;
        }

        if (Hdf5Arrays.IsTensor(value))
        {
            parent[name] = Hdf5Arrays.ToDataset(value, compression, path);
            return;
        }

        if (value is IReadOnlyDictionary<string, object?> map)
        {
            if (map.ContainsKey(VersionableEnvelope.WrappedKey))
            {
                WriteNested(parent, name, map, compression, path);
            }
            else
            {
                WriteMap(parent, name, map, shape, compression, path);
            }

            return;
        }

        // A converter's output may itself be a list — complex is [real, imaginary] — and Python
        // writes that as an attribute, because its converter arm runs before its container arm.
        // The declared type is the only thing that tells the two apart once the value is lowered.
        if (value is IReadOnlyList<object?> list && shape.Kind != Hdf5ShapeKind.Scalar)
        {
            WriteSequence(parent, name, list, shape, compression, path);
            return;
        }

        parent.Attributes[name] = Hdf5Values.ToAttribute(value, path);
    }

    private static void WriteNested(
        H5Group parent,
        string name,
        IReadOnlyDictionary<string, object?> wire,
        Hdf5Compression compression,
        string path)
    {
        EnvelopeMetadata envelope = EnvelopeCodec.Read(wire);
        VersionableMetadata? metadata = envelope.ObjectName is not null
            && VersionableRegistry.TryGetByName(envelope.ObjectName, out VersionableMetadata? found)
            ? found
            : null;

        H5Group subgroup = [];
        WriteObject(subgroup, EnvelopeCodec.Strip(wire), envelope, metadata, compression, path);
        parent[name] = subgroup;
    }

    private static void WriteMap(
        H5Group parent,
        string name,
        IReadOnlyDictionary<string, object?> map,
        Hdf5TypeShape shape,
        Hdf5Compression compression,
        string path)
    {
        Hdf5TypeShape valueShape = shape.Kind == Hdf5ShapeKind.Map ? shape.Element : Hdf5TypeShape.Unknown;
        H5Group subgroup = [];
        foreach (KeyValuePair<string, object?> entry in map)
        {
            WriteValue(
                subgroup,
                Hdf5Values.EncodeName(entry.Key, path),
                entry.Value,
                valueShape,
                compression,
                $"{Describe(path)}[{entry.Key}]");
        }

        parent[name] = subgroup;
    }

    private static void WriteSequence(
        H5Group parent,
        string name,
        IReadOnlyList<object?> values,
        Hdf5TypeShape shape,
        Hdf5Compression compression,
        string path)
    {
        Type? elementType = shape.Kind == Hdf5ShapeKind.Sequence ? shape.ElementType : null;

        if (Hdf5TypeShape.IsDatasetElement(elementType)
            || (elementType is null && values.All(IsDatasetScalar)))
        {
            parent[name] = Hdf5Values.ToScalarDataset(values, elementType, path);
            return;
        }

        Hdf5TypeShape elementShape = Hdf5TypeShape.Of(elementType);
        H5Group subgroup = [];
        for (int index = 0; index < values.Count; index++)
        {
            WriteValue(
                subgroup,
                index.ToString(CultureInfo.InvariantCulture),
                values[index],
                elementShape,
                compression,
                $"{Describe(path)}[{index}]");
        }

        parent[name] = subgroup;
    }

    private static bool IsDatasetScalar(object? value) =>
        value is string or bool or sbyte or short or int or long or byte or ushort or uint or ulong
            or Half or float or double;

    private static string Join(string path, string name) => path.Length == 0 ? name : $"{path}.{name}";

    private static string Describe(string path) => path.Length == 0 ? "<root>" : path;
}
