using System.Globalization;
using PureHDF;
using PureHDF.VOL.Native;
using Versionable.Engine;
using Versionable.Errors;

namespace Versionable.Backends.Hdf5;

/// <summary>
/// Turns an HDF5 file back into the wire values the engine materializes from.
/// </summary>
/// <remarks>
/// Python counterpart: <c>_readMeta</c>, <c>_readFields</c>, <c>_readGroup</c>,
/// <c>_readSequenceGroup</c>, <c>_readDictGroup</c>, and <c>_readVersionableGroup</c> in
/// <c>src/versionable/_hdf5_backend.py</c>.
/// <para>
/// <b>Reading dispatches on the declared type, as Python's does.</b> A group of integer-named
/// children is a list or a dictionary depending only on the annotation, and a 1-D dataset is a
/// <c>list[float]</c> or a <c>Tensor&lt;double&gt;</c> for the same reason. Where no
/// annotation is available — an unknown field kept by <c>Preserve</c>, or the metadata-only
/// probe — the fallbacks are Python's: a group reads as a dictionary and a dataset as a list.
/// </para>
/// </remarks>
internal static class Hdf5Reader
{
    /// <summary>Reads one file.</summary>
    /// <param name="path">Input path.</param>
    /// <param name="options">Which fields to materialize, and the target type's metadata.</param>
    /// <returns>The wire values and the envelope.</returns>
    /// <exception cref="BackendException">The read failed or the file is malformed.</exception>
    internal static BackendLoadResult Read(string path, BackendLoadOptions options)
    {
        NativeFile file;
        try
        {
            file = H5File.OpenRead(path);
        }
        catch (Exception error) when (error is not VersionableException)
        {
            throw new BackendException($"Failed to read HDF5 from '{path}': {error.Message}", error);
        }

        using (file)
        {
            HashSet<string> lazy = new(StringComparer.Ordinal);
            Dictionary<string, object?> document;
            try
            {
                document = ReadObject(
                    file, options.TargetMetadata, options.MetadataOnly, new Hdf5Skips(lazy), string.Empty);
            }
            catch (Exception error) when (Hdf5Diagnostics.FilterHint(error).Length > 0)
            {
                // A filter the file uses and this build does not register surfaces here rather
                // than at open, because HDF5 only runs the pipeline when a chunk is read. Python
                // appends the same note from the same place (`missingFilterHint`).
                throw new BackendException(
                    $"Failed to read HDF5 from '{path}': {error.Message}{Hdf5Diagnostics.FilterHint(error)}",
                    error);
            }

            EnvelopeMetadata envelope = EnvelopeCodec.Read(document);
            return new BackendLoadResult(EnvelopeCodec.Strip(document), envelope, lazy);
        }
    }

    /// <summary>Reads a group as an object: its envelope table plus its fields.</summary>
    /// <remarks>
    /// The <c>__versionable__</c> child comes back under the same key it has in every text
    /// backend, which is what lets <see cref="EnvelopeCodec"/> and
    /// <c>Engine.ObjectMaterializer</c> read a nested HDF5 object with no HDF5 knowledge at all.
    /// </remarks>
    private static Dictionary<string, object?> ReadObject(
        IH5Group group,
        VersionableMetadata? metadata,
        bool metadataOnly,
        Hdf5Skips lazy,
        string path)
    {
        Dictionary<string, Type?> declared = new(StringComparer.Ordinal);
        if (metadata is not null)
        {
            foreach (FieldDescriptor field in metadata.Fields)
            {
                declared[field.WireName] = field.ClrType;
            }
        }

        Dictionary<string, object?> fields = new(StringComparer.Ordinal);

        foreach (IH5Object child in group.Children())
        {
            string name = child.Name;
            if (name == VersionableEnvelope.WrappedKey)
            {
                fields[name] = child is IH5Group meta
                    ? ReadAttributes(meta, decodeNames: false, Join(path, name))
                    : null;
                continue;
            }

            Hdf5TypeShape shape = Hdf5TypeShape.Of(declared.GetValueOrDefault(name));
            string childPath = Join(path, name);

            if (metadataOnly && IsArrayData(child, shape))
            {
                lazy.Skip(name);
                continue;
            }

            fields[name] = ReadChild(child, shape, metadataOnly, lazy.Into(name), childPath);
        }

        foreach (KeyValuePair<string, object?> attribute in ReadAttributes(group, decodeNames: false, path))
        {
            fields[attribute.Key] = attribute.Value;
        }

        return fields;
    }

    private static object? ReadChild(
        IH5Object child,
        Hdf5TypeShape shape,
        bool metadataOnly,
        Hdf5Skips lazy,
        string path) =>
        child switch
        {
            IH5Dataset dataset => shape.Kind == Hdf5ShapeKind.Tensor && shape.ElementType is not null
                ? Hdf5Arrays.ReadTensor(dataset, shape.ElementType, path)
                : Hdf5Arrays.ReadValues(dataset, path),
            IH5Group group => ReadGroup(group, shape, metadataOnly, lazy, path),
            _ => null,
        };

    private static object ReadGroup(
        IH5Group group,
        Hdf5TypeShape shape,
        bool metadataOnly,
        Hdf5Skips lazy,
        string path)
    {
        if (group.LinkExists(VersionableEnvelope.WrappedKey))
        {
            return ReadNested(group, shape, metadataOnly, lazy, path);
        }

        return shape.Kind switch
        {
            Hdf5ShapeKind.Sequence => ReadSequence(group, shape.Element, metadataOnly, lazy, path),
            // Python's `_readGroup` falls back to a dictionary for anything it cannot place,
            // and so does this: a group whose annotation is missing is far more often a dict of
            // named things than a list of numbered ones.
            _ => ReadMap(group, shape.Kind == Hdf5ShapeKind.Map ? shape.Element : Hdf5TypeShape.Unknown,
                metadataOnly, lazy, path),
        };
    }

    private static Dictionary<string, object?> ReadNested(
        IH5Group group,
        Hdf5TypeShape shape,
        bool metadataOnly,
        Hdf5Skips lazy,
        string path)
    {
        // The envelope names the concrete type; the declared type is only its upper bound. That
        // order matters for a polymorphic collection: a subclass may add a container field the
        // base does not declare, and a group read against the base would have no annotation for
        // it and fall back to reading it as a dictionary. The declared type is the fallback,
        // which is what still resolves a `Register = false` type — absent from the name index.
        VersionableMetadata? metadata = null;
        string envelopeName = ReadObjectName(group, path);

        if (envelopeName.Length > 0)
        {
            VersionableRegistry.TryGetByName(envelopeName, out metadata);
        }

        if (metadata is null && shape.DeclaredType is not null)
        {
            VersionableRegistry.TryGetByType(shape.DeclaredType, out metadata);
        }

        // The skip recorder descends with the object, so a nested type's skipped array is
        // recorded as `inner/values` rather than dropped. Passing null here was the bug: the
        // field vanished from `Fields` with nothing saying why, and the engine could only
        // report it as missing from the file.
        return ReadObject(group, metadata, metadataOnly, lazy, path);
    }

    /// <summary>Reads only the Serialization Name out of a group's metadata child.</summary>
    /// <returns>The name, or the empty string when the group records none.</returns>
    private static string ReadObjectName(IH5Group group, string path)
    {
        if (group.Get(VersionableEnvelope.WrappedKey) is not IH5Group meta
            || !meta.AttributeExists(VersionableEnvelope.ObjectKey))
        {
            return string.Empty;
        }

        return Hdf5Values.FromAttribute(meta.Attribute(VersionableEnvelope.ObjectKey), path) as string
            ?? string.Empty;
    }

    private static List<object?> ReadSequence(
        IH5Group group,
        Hdf5TypeShape elementShape,
        bool metadataOnly,
        Hdf5Skips lazy,
        string path)
    {
        Dictionary<string, IH5Object> children = new(StringComparer.Ordinal);
        foreach (IH5Object child in group.Children())
        {
            children[child.Name] = child;
        }

        Dictionary<string, object?> attributes = ReadAttributes(group, decodeNames: false, path);

        List<string> keys = [.. children.Keys.Concat(attributes.Keys).Distinct(StringComparer.Ordinal)];
        keys.Sort(CompareIndexNames);

        List<object?> values = new(keys.Count);
        foreach (string key in keys)
        {
            string elementPath = $"{Describe(path)}[{key}]";
            // The element descends with the *container's* skip prefix, not one of its own: an
            // index is not part of a skip's identity. Every element of a container has one
            // declared type, so a field skipped in one is skipped in all, and recording it once
            // per index would make the engine match paths it can never reconstruct — a generated
            // container reader hands the engine an element, never which element.
            values.Add(children.TryGetValue(key, out IH5Object? child)
                ? ReadChild(child, elementShape, metadataOnly, lazy, elementPath)
                : attributes[key]);
        }

        return values;
    }

    private static Dictionary<string, object?> ReadMap(
        IH5Group group,
        Hdf5TypeShape valueShape,
        bool metadataOnly,
        Hdf5Skips lazy,
        string path)
    {
        Dictionary<string, object?> entries = new(StringComparer.Ordinal);

        foreach (IH5Object child in group.Children())
        {
            if (child.Name == VersionableEnvelope.WrappedKey)
            {
                continue;
            }

            string key = Hdf5Values.DecodeName(child.Name);
            // Values descend with the container's prefix; see the note in ReadSequence.
            entries[key] = ReadChild(child, valueShape, metadataOnly, lazy, $"{Describe(path)}[{key}]");
        }

        foreach (KeyValuePair<string, object?> attribute in ReadAttributes(group, decodeNames: true, path))
        {
            entries[attribute.Key] = attribute.Value;
        }

        return entries;
    }

    private static Dictionary<string, object?> ReadAttributes(IH5Group group, bool decodeNames, string path)
    {
        Dictionary<string, object?> values = new(StringComparer.Ordinal);
        foreach (IH5Attribute attribute in group.Attributes())
        {
            string key = decodeNames ? Hdf5Values.DecodeName(attribute.Name) : attribute.Name;
            values[key] = Hdf5Values.FromAttribute(attribute, $"{Describe(path)}.{key}");
        }

        return values;
    }

    /// <summary>
    /// Whether a child holds array data, and so is what <c>MetadataOnly</c> skips.
    /// </summary>
    /// <remarks>
    /// Python counterpart: <c>_isArrayField</c> and <c>_isArrayCollectionField</c>, including
    /// the case that looks like an oversight and is not: <c>_isArrayField(None)</c> returns
    /// <c>True</c>. An <em>unannotated dataset</em> is assumed to be array data, which is what
    /// makes the metadata-only probe cheap before the file has named its type — and
    /// <c>VersionableFile.LoadDynamic(path)</c> probes exactly that way, with no
    /// <see cref="BackendLoadOptions.TargetMetadata"/> to consult. An unannotated <em>group</em>
    /// is not assumed to be anything and is read, as Python reads it.
    /// <para>
    /// A <c>list[float]</c> is a dataset too, and stays eager: it is not array data, and the
    /// cost this mode exists to avoid is the array's.
    /// </para>
    /// </remarks>
    private static bool IsArrayData(IH5Object child, Hdf5TypeShape shape) => child switch
    {
        IH5Dataset => shape.Kind is Hdf5ShapeKind.Tensor or Hdf5ShapeKind.Unknown,
        IH5Group => shape.Kind is Hdf5ShapeKind.Sequence or Hdf5ShapeKind.Map
            && shape.Element.Kind == Hdf5ShapeKind.Tensor,
        _ => false,
    };

    /// <summary>Orders the integer-named children of a sequence group.</summary>
    /// <remarks>
    /// Python sorts with <c>key=int</c>, which would raise on a non-numeric name. Falling back
    /// to an ordinal comparison instead keeps a hand-edited file readable, and a group whose
    /// names are all numeric — every group this writer produces — sorts identically.
    /// </remarks>
    private static int CompareIndexNames(string left, string right)
    {
        bool leftIsIndex = long.TryParse(left, NumberStyles.Integer, CultureInfo.InvariantCulture, out long leftIndex);
        bool rightIsIndex = long.TryParse(
            right, NumberStyles.Integer, CultureInfo.InvariantCulture, out long rightIndex);

        return leftIsIndex && rightIsIndex
            ? leftIndex.CompareTo(rightIndex)
            : string.CompareOrdinal(left, right);
    }

    private static string Join(string path, string name) => path.Length == 0 ? name : $"{path}.{name}";

    private static string Describe(string path) => path.Length == 0 ? "<root>" : path;
}
