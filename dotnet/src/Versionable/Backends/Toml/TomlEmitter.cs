using System.Text;
using Tomlyn.Helpers;
using Tomlyn.Model;
using Tomlyn.Serialization;

namespace Versionable.Backends.Toml;

/// <summary>
/// Renders a wire dictionary as TOML text: key/value lines, table headers, arrays of tables, and
/// the commented-out defaults <c>commentDefaults</c> asks for.
/// </summary>
/// <remarks>
/// Python counterpart: <c>tomlkit.dumps</c> for the ordinary path and
/// <c>_emitWithCommentedDefaults</c> / <c>_addContainerWithDefaults</c> for the commented one, in
/// <c>src/versionable/_toml_backend.py</c>. Both Python paths are one method here, because the
/// second is the first plus a predicate.
/// <para>
/// <b>Why an emitter rather than Tomlyn's writer.</b> <see cref="TomlWriter"/> renders every
/// array of tables inline — <c>points = [{x = 0.0}, {x = 1.0}]</c> — with no option to switch it
/// to <c>[[points]]</c> headers, and it buffers a whole document so it can hoist scalars above
/// tables, which leaves nowhere to interleave a comment line. Both are load-bearing:
/// <c>conformance/golden/nested/nested.toml</c> is written in header form, and a commented
/// default is a comment line in the middle of a table. So the structure is emitted here and every
/// <em>value</em> is still rendered by <see cref="TomlWriter"/> — one scratch document per
/// key/value line — which keeps string escaping, float formatting, and <c>nan</c>/<c>inf</c>
/// spelling Tomlyn's rather than this file's.
/// </para>
/// <para>
/// <b>Ordering</b> reproduces tomlkit's, because TOML requires it: a bare <c>key = value</c> after
/// a <c>[table]</c> header belongs to that table, so every level emits its scalars first, then its
/// sub-tables. Within those two groups the order is the caller's — which puts
/// <c>[__versionable__]</c> first among the tables, since the backend inserts it first.
/// </para>
/// </remarks>
internal sealed class TomlEmitter
{
    private readonly StringBuilder _builder = new();
    private readonly IReadOnlySet<Type> _nativeTypes;
    private readonly bool _commentDefaults;

    private TomlEmitter(bool commentDefaults, IReadOnlySet<Type> nativeTypes)
    {
        _commentDefaults = commentDefaults;
        _nativeTypes = nativeTypes;
    }

    /// <summary>Renders a whole document.</summary>
    /// <param name="root">The root table: envelope first, then TOML-safe field values.</param>
    /// <param name="metadata">
    /// Metadata of the type being written, which supplies the defaults <paramref name="commentDefaults"/>
    /// compares against. Ignored when <paramref name="commentDefaults"/> is <see langword="false"/>.
    /// </param>
    /// <param name="commentDefaults">Write fields still at their default as comment lines.</param>
    /// <param name="nativeTypes">The backend's native types, for lowering default values.</param>
    /// <returns>The document text, LF-terminated.</returns>
    internal static string Render(
        IReadOnlyDictionary<string, object?> root,
        VersionableMetadata? metadata,
        bool commentDefaults,
        IReadOnlySet<Type> nativeTypes)
    {
        TomlEmitter emitter = new(commentDefaults, nativeTypes);
        emitter.WriteContainer([], root, metadata);
        return emitter._builder.ToString();
    }

    /// <summary>Renders <c>key = value</c> exactly as Tomlyn's writer would.</summary>
    /// <param name="key">The key, quoted by Tomlyn if it is not bare-safe.</param>
    /// <param name="value">A TOML-safe wire value that is not a section.</param>
    /// <returns>One LF-terminated line.</returns>
    internal static string RenderKeyValue(string key, object? value)
    {
        StringWriter text = new() { NewLine = "\n" };
        TomlWriter writer = new(text, TomlWire.Options);

        writer.WriteStartDocument();
        writer.WriteStartTable();
        writer.WritePropertyName(key);
        TomlWire.WriteValue(writer, value);
        writer.WriteEndTable();
        writer.WriteEndDocument();

        return text.ToString();
    }

    /// <summary>
    /// Whether a value becomes its own <c>[section]</c> rather than a <c>key = value</c> line.
    /// </summary>
    /// <param name="value">A TOML-safe wire value.</param>
    /// <returns><see langword="true"/> for a table or a non-empty array of tables.</returns>
    private static bool IsSection(object? value) =>
        value is IReadOnlyDictionary<string, object?> || IsTableArray(value);

    private static bool IsTableArray(object? value) =>
        value is IReadOnlyList<object?> { Count: > 0 } items
            && items.All(static item => item is IReadOnlyDictionary<string, object?>);

    /// <summary>Renders one key of a table header path.</summary>
    /// <param name="key">The key to render.</param>
    /// <returns>The key bare when TOML allows it, otherwise as a quoted basic string.</returns>
    private static string FormatKey(string key)
    {
        if (key.Length == 0)
        {
            return TomlFormatHelper.ToString(key, TomlPropertyDisplayKind.Default);
        }

        foreach (char character in key)
        {
            // TOML 1.0 §Keys: a bare key is ASCII letters, digits, underscores, and dashes.
            bool bare = character is (>= 'A' and <= 'Z') or (>= 'a' and <= 'z') or (>= '0' and <= '9')
                or '_' or '-';
            if (!bare)
            {
                return TomlFormatHelper.ToString(key, TomlPropertyDisplayKind.Default);
            }
        }

        return key;
    }

    private static string FormatPath(IReadOnlyList<string> path) =>
        string.Join('.', path.Select(FormatKey));

    private void WriteContainer(
        IReadOnlyList<string> path,
        IReadOnlyDictionary<string, object?> data,
        VersionableMetadata? metadata)
    {
        IReadOnlyDictionary<string, object?> defaults = _commentDefaults && metadata is not null
            ? TomlDefaults.For(metadata, _nativeTypes)
            : TomlDefaults.None;

        // Pass 1: the key/value lines, with an at-default value commented out instead of written.
        //
        // Sections are noted here and emitted in pass 3, commented or not. TOML's scalars-before-
        // tables rule has to hold for commented text as well as live text, because a comment is
        // only worth writing if deleting the `#` gives a valid file: a `key = value` line printed
        // after a commented `[section]` block reads as a live key of that block's *last* table the
        // moment the block is uncommented. For a defaulted nested object that table is its
        // `__versionable__`, so the value would be swallowed as an unknown envelope key and the
        // field would silently fall back to its default. Emitting every section after every scalar
        // is what keeps both the file and its uncommented form saying the same thing.
        HashSet<string> commented = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, object?> entry in data)
        {
            if (entry.Key == VersionableEnvelope.WrappedKey)
            {
                continue;
            }

            bool atDefault = defaults.TryGetValue(entry.Key, out object? fallback)
                && TomlDefaults.WireEquals(entry.Value, fallback);
            if (atDefault)
            {
                commented.Add(entry.Key);
            }

            if (IsSection(entry.Value))
            {
                continue;
            }

            if (atDefault)
            {
                WriteCommented(path, entry.Key, entry.Value);
            }
            else
            {
                _builder.Append(RenderKeyValue(entry.Key, entry.Value));
            }
        }

        // Pass 2: the envelope, never commented and always the first table at its level, so a
        // reader sees what the file is before it sees what is in it.
        if (data.TryGetValue(VersionableEnvelope.WrappedKey, out object? envelope)
            && envelope is IReadOnlyDictionary<string, object?> envelopeTable)
        {
            WriteTable([.. path, VersionableEnvelope.WrappedKey], envelopeTable, metadata: null);
        }

        // Pass 3: sections, live and commented alike, in declaration order.
        foreach (KeyValuePair<string, object?> entry in data)
        {
            if (entry.Key == VersionableEnvelope.WrappedKey || !IsSection(entry.Value))
            {
                continue;
            }

            string[] childPath = [.. path, entry.Key];
            if (commented.Contains(entry.Key))
            {
                WriteCommented(path, entry.Key, entry.Value);
            }
            else if (entry.Value is IReadOnlyDictionary<string, object?> table)
            {
                WriteTable(childPath, table, NestedMetadata(metadata, entry.Key));
            }
            else if (IsTableArray(entry.Value))
            {
                foreach (object? element in (IReadOnlyList<object?>)entry.Value!)
                {
                    WriteHeader($"[[{FormatPath(childPath)}]]");
                    WriteContainer(childPath, (IReadOnlyDictionary<string, object?>)element!, metadata: null);
                }
            }
        }
    }

    private void WriteTable(
        IReadOnlyList<string> path,
        IReadOnlyDictionary<string, object?> table,
        VersionableMetadata? metadata)
    {
        // A table holding nothing but other tables gets no header of its own — `[byName.origin]`
        // implies `byName`. That is tomlkit's super-table rendering and it is what the golden
        // corpus holds; an empty table is the one case that does need a header, because nothing
        // else would record that it exists.
        bool needsHeader = table.Count == 0;
        foreach (KeyValuePair<string, object?> entry in table)
        {
            if (!IsSection(entry.Value))
            {
                needsHeader = true;
                break;
            }
        }

        if (needsHeader)
        {
            WriteHeader($"[{FormatPath(path)}]");
        }

        WriteContainer(path, table, metadata);
    }

    private void WriteCommented(IReadOnlyList<string> path, string key, object? value)
    {
        string rendered;
        if (IsSection(value))
        {
            // A default that renders as a section is several lines, and every one of them has to
            // carry a `#`: comment only the header and the keys under it would land in whatever
            // table happens to precede them. The block is rendered by a fresh emitter with
            // commenting off, because nothing inside a block that is entirely absent from the
            // object can itself be at-or-off its default.
            //
            // The header path is the full dotted one, so uncommenting the block writes to the
            // field it came from. Python renders the block in isolation and so emits a
            // root-relative header, which would land the values in the wrong table for anything
            // nested — a deliberate divergence, recorded on BackendSaveOptions.CommentDefaults.
            TomlEmitter block = new(commentDefaults: false, _nativeTypes);
            string[] childPath = [.. path, key];
            if (value is IReadOnlyDictionary<string, object?> table)
            {
                block.WriteTable(childPath, table, metadata: null);
            }
            else
            {
                foreach (object? element in (IReadOnlyList<object?>)value!)
                {
                    block.WriteHeader($"[[{FormatPath(childPath)}]]");
                    block.WriteContainer(childPath, (IReadOnlyDictionary<string, object?>)element!, metadata: null);
                }
            }

            rendered = block._builder.ToString();

            // Set off from what precedes it exactly as a live section is by WriteHeader, so the
            // commented block reads as a block and not as a footnote to the table above it.
            if (_builder.Length > 0)
            {
                _builder.Append('\n');
            }
        }
        else
        {
            rendered = RenderKeyValue(key, value);
        }

        foreach (string line in rendered.Split('\n'))
        {
            if (line.Length > 0)
            {
                _builder.Append("# ").Append(line).Append('\n');
            }
        }
    }

    private void WriteHeader(string header)
    {
        if (_builder.Length > 0)
        {
            _builder.Append('\n');
        }

        _builder.Append(header).Append('\n');
    }

    private static VersionableMetadata? NestedMetadata(VersionableMetadata? metadata, string wireName)
    {
        if (metadata is null)
        {
            return null;
        }

        foreach (FieldDescriptor field in metadata.Fields)
        {
            if (field.WireName == wireName)
            {
                // Only a field declared as a [Versionable] type recurses. A `List<Inner>` field
                // does not, matching Python's `_findVersionableType`: the elements of a container
                // have no declared default of their own to compare against.
                return VersionableRegistry.TryGetByType(field.ClrType, out VersionableMetadata? nested)
                    ? nested
                    : null;
            }
        }

        return null;
    }
}
