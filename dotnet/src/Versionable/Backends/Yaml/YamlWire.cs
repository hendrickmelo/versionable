using System.Globalization;
using Versionable.Errors;
using YamlDotNet.Core;
using YamlDotNet.Core.Events;

namespace Versionable.Backends.Yaml;

/// <summary>
/// Moves between YamlDotNet's event stream and the plain CLR primitives, lists, and dictionaries
/// the engine's walker speaks.
/// </summary>
/// <remarks>
/// Written against <see cref="IParser"/> and <see cref="IEmitter"/> — the event layer — rather
/// than YamlDotNet's <c>Deserializer</c>/<c>Serializer</c>, for the same three reasons the JSON
/// backend is hand-written plus one of its own. The object serializers reflect over runtime types,
/// which <c>IsAotCompatible</c> rejects; they would let a CLR type the walker never lowered reach
/// a file unnoticed; and their type resolution is <em>YamlDotNet's</em> schema, not PyYAML's — it
/// would read <c>yes</c> as the string "yes" and <c>2026-08-05</c> as a
/// <see cref="System.DateTime"/>, disagreeing with Python about files Python wrote. At the event
/// layer every scalar arrives as text with its quoting style, and <see cref="YamlSchema"/> decides
/// what it means.
/// </remarks>
internal static class YamlWire
{
    // Matches JsonDocument's default max depth, and exists for the same reason: recursion here is
    // driven by the file, and a document nested a few hundred thousand levels deep would overflow
    // the stack, which .NET cannot catch. No real schema comes close.
    private const int _maxDepth = 64;

    /// <summary>Reads a whole document into engine-shaped CLR values.</summary>
    /// <param name="parser">A parser positioned at the start of the stream.</param>
    /// <param name="path">File path, for error messages.</param>
    /// <returns>
    /// The document's root mapping, or <see langword="null"/> for an empty file. Values are
    /// <see cref="string"/>, <see cref="bool"/>, <see cref="long"/>, <see cref="ulong"/>,
    /// <see cref="double"/>, <see cref="Dictionary{TKey, TValue}"/>, <see cref="List{T}"/>, or
    /// <see langword="null"/>.
    /// </returns>
    /// <exception cref="BackendException">The stream is not a single mapping document.</exception>
    internal static Dictionary<string, object?>? ReadDocument(IParser parser, string path)
    {
        // Driven off the raw event cursor rather than YamlDotNet's Consume/TryConsume extensions.
        // Those leave their out parameter unannotated for nullability, so every call site would
        // need a null-forgiving operator to compile under `Nullable=enable` — which is exactly the
        // annotation this reader should not be sprinkling around. The contract below is explicit
        // instead: ReadNode is entered with Current on the node's first event and returns with
        // Current on its last, so a caller always knows whose turn it is to advance.
        if (Advance(parser, path) is not StreamStart)
        {
            throw new BackendException($"'{path}' does not begin a YAML stream.");
        }

        if (Advance(parser, path) is StreamEnd)
        {
            return null;
        }

        if (parser.Current is not DocumentStart)
        {
            throw new BackendException($"'{path}' does not begin a YAML document.");
        }

        Advance(parser, path);
        Dictionary<AnchorName, object?> anchors = [];
        object? root = ReadNode(parser, anchors, path, depth: 0);

        if (Advance(parser, path) is not DocumentEnd || Advance(parser, path) is not StreamEnd)
        {
            // Python counterpart: `yaml.safe_load` on a multi-document stream raises
            // ComposerError. Silently taking the first document would drop data the file holds.
            throw new BackendException(
                $"'{path}' holds more than one YAML document. A versionable file is a single "
                    + "mapping.");
        }

        if (root is null)
        {
            return null;
        }

        return root as Dictionary<string, object?>
            ?? throw new BackendException(
                $"Expected a YAML mapping in '{path}', found {DescribeKind(root)}.");
    }

    /// <summary>Writes an engine-shaped wire value.</summary>
    /// <param name="emitter">The emitter positioned to accept a value.</param>
    /// <param name="value">A value produced by the engine walker.</param>
    /// <exception cref="BackendException">The value is not something YAML can hold.</exception>
    internal static void Write(IEmitter emitter, object? value)
    {
        switch (value)
        {
            case null:
                WriteScalar(emitter, "null", ScalarStyle.Plain);
                break;
            case string text:
                // Any lets the emitter's own analysis pick plain or quoted for structural reasons
                // (a leading `-`, an embedded `: `, a trailing space). SingleQuoted overrides it
                // where the text would otherwise be read back as a number, a bool, a date, or
                // null — which the emitter has no idea about, since that is PyYAML's schema and
                // not the spec's.
                WriteScalar(
                    emitter,
                    text,
                    YamlSchema.IsPlainSafe(text) ? ScalarStyle.Any : ScalarStyle.SingleQuoted);
                break;
            case bool flag:
                WriteScalar(emitter, flag ? "true" : "false", ScalarStyle.Plain);
                break;
            case sbyte or short or int or long:
                WriteScalar(
                    emitter,
                    Convert.ToInt64(value, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture),
                    ScalarStyle.Plain);
                break;
            case byte or ushort or uint or ulong:
                WriteScalar(
                    emitter,
                    Convert.ToUInt64(value, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture),
                    ScalarStyle.Plain);
                break;
            case float single:
                WriteScalar(emitter, YamlSchema.RenderReal(single), ScalarStyle.Plain);
                break;
            case double real:
                WriteScalar(emitter, YamlSchema.RenderReal(real), ScalarStyle.Plain);
                break;
            case decimal money:
                WriteScalar(emitter, YamlSchema.RenderReal(money), ScalarStyle.Plain);
                break;
            case IReadOnlyDictionary<string, object?> map:
                WriteMapping(emitter, map);
                break;
            case IEnumerable<object?> sequence:
                emitter.Emit(new SequenceStart(AnchorName.Empty, TagName.Empty, true, SequenceStyle.Block));
                foreach (object? item in sequence)
                {
                    Write(emitter, item);
                }

                emitter.Emit(new SequenceEnd());
                break;
            default:
                throw new BackendException(
                    $"The YAML backend cannot write '{value.GetType()}'. The walker should have "
                        + "lowered it to a primitive, list, or dictionary first.");
        }
    }

    /// <summary>Writes a dictionary as a block mapping.</summary>
    /// <param name="emitter">The emitter positioned to accept a value.</param>
    /// <param name="map">Entries to write, in enumeration order.</param>
    internal static void WriteMapping(IEmitter emitter, IReadOnlyDictionary<string, object?> map)
    {
        emitter.Emit(new MappingStart(AnchorName.Empty, TagName.Empty, true, MappingStyle.Block));
        foreach (KeyValuePair<string, object?> entry in map)
        {
            Write(emitter, entry.Key);
            Write(emitter, entry.Value);
        }

        emitter.Emit(new MappingEnd());
    }

    /// <summary>
    /// Emits a scalar with both implicit flags set, so no tag is written whichever style wins.
    /// </summary>
    private static void WriteScalar(IEmitter emitter, string text, ScalarStyle style) =>
        emitter.Emit(new Scalar(AnchorName.Empty, TagName.Empty, text, style, true, true));

    /// <summary>
    /// Reads one node. Entered with <c>parser.Current</c> on the node's first event; returns with
    /// it on the node's last.
    /// </summary>
    private static object? ReadNode(
        IParser parser,
        Dictionary<AnchorName, object?> anchors,
        string path,
        int depth)
    {
        if (depth > _maxDepth)
        {
            throw new BackendException(
                $"'{path}' nests collections more than {_maxDepth} deep, which no schema does and "
                    + "which a reader cannot follow without risking the stack.");
        }

        switch (parser.Current)
        {
            case AnchorAlias alias:
                return anchors.TryGetValue(alias.Value, out object? anchored)
                    ? anchored
                    : throw new BackendException($"'{path}' refers to the undefined anchor '*{alias.Value}'.");

            case Scalar scalar:
                object? value = ReadScalar(scalar);
                Remember(anchors, scalar.Anchor, value);
                return value;

            case SequenceStart sequenceStart:
                List<object?> items = [];

                // Registered before the elements are read, so a sequence containing an alias to
                // itself resolves instead of failing. PyYAML's constructors do the same, by
                // yielding the empty container before filling it.
                Remember(anchors, sequenceStart.Anchor, items);
                while (Advance(parser, path) is not SequenceEnd)
                {
                    items.Add(ReadNode(parser, anchors, path, depth + 1));
                }

                return items;

            case MappingStart mappingStart:
                Dictionary<string, object?> map = new(StringComparer.Ordinal);
                Remember(anchors, mappingStart.Anchor, map);
                ReadMappingEntries(parser, anchors, path, map, depth);
                return map;

            default:
                throw new BackendException(
                    $"Unexpected YAML event {parser.Current?.GetType().Name} in '{path}'.");
        }
    }

    private static void ReadMappingEntries(
        IParser parser,
        Dictionary<AnchorName, object?> anchors,
        string path,
        Dictionary<string, object?> map,
        int depth)
    {
        // Merge keys are gathered and applied first so that explicitly written keys overwrite what
        // an anchor supplied, which is what Python's `flatten_mapping` arranges by placing the
        // merged pairs ahead of the node's own.
        List<KeyValuePair<string, object?>> merged = [];
        List<KeyValuePair<string, object?>> own = [];

        while (Advance(parser, path) is not MappingEnd)
        {
            if (parser.Current is not Scalar key)
            {
                throw new BackendException(
                    $"'{path}' has a non-scalar mapping key, which versionable cannot represent: "
                        + "every key becomes a field name or a dictionary key string.");
            }

            Advance(parser, path);
            object? value = ReadNode(parser, anchors, path, depth + 1);

            if (key.Style == ScalarStyle.Plain && key.Value == YamlSchema.MergeKey)
            {
                CollectMerge(value, merged, path);
                continue;
            }

            // The key's text, not its resolved value: the writer produced these keys from strings
            // (the walker stringifies every dictionary key before it reaches a backend), so the
            // text is what round-trips, and a declared `dict[int, str]` gets its keys back through
            // the generated reader either way.
            own.Add(new KeyValuePair<string, object?>(key.Value, value));
        }

        foreach (KeyValuePair<string, object?> entry in merged)
        {
            map[entry.Key] = entry.Value;
        }

        foreach (KeyValuePair<string, object?> entry in own)
        {
            map[entry.Key] = entry.Value;
        }
    }

    private static void CollectMerge(
        object? value,
        List<KeyValuePair<string, object?>> merged,
        string path)
    {
        switch (value)
        {
            case Dictionary<string, object?> single:
                merged.AddRange(single);
                break;
            case List<object?> sequence:
                // Reversed, as Python does: earlier entries of the list win, and since later
                // additions overwrite earlier ones here, the first has to be applied last.
                for (int index = sequence.Count - 1; index >= 0; index--)
                {
                    if (sequence[index] is not Dictionary<string, object?> item)
                    {
                        throw new BackendException(
                            $"'{path}' merges a non-mapping ({DescribeKind(sequence[index])}) into a mapping.");
                    }

                    merged.AddRange(item);
                }

                break;
            default:
                throw new BackendException(
                    $"'{path}' merges a non-mapping ({DescribeKind(value)}) into a mapping.");
        }
    }

    private static object? ReadScalar(Scalar scalar)
    {
        // A quoted, literal, or folded scalar is a string by construction — that is what quoting
        // means — so only a plain one is offered to the schema.
        if (scalar.Style != ScalarStyle.Plain)
        {
            return scalar.Value;
        }

        if (!scalar.Tag.IsEmpty && !scalar.Tag.IsNonSpecific && scalar.Tag.Value == "tag:yaml.org,2002:str")
        {
            return scalar.Value;
        }

        return YamlSchema.TryResolveImplicit(scalar.Value, out object? value) ? value : scalar.Value;
    }

    /// <summary>Moves to the next event and returns it.</summary>
    /// <exception cref="BackendException">The stream ended mid-document.</exception>
    private static ParsingEvent Advance(IParser parser, string path) =>
        parser.MoveNext() && parser.Current is not null
            ? parser.Current
            : throw new BackendException($"'{path}' ends in the middle of a YAML document.");

    private static void Remember(Dictionary<AnchorName, object?> anchors, AnchorName anchor, object? value)
    {
        if (!anchor.IsEmpty)
        {
            anchors[anchor] = value;
        }
    }

    private static string DescribeKind(object? value) =>
        value switch
        {
            null => "null",
            Dictionary<string, object?> => "a mapping",
            List<object?> => "a sequence",
            _ => $"the scalar {value}",
        };
}
