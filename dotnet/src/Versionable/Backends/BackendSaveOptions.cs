namespace Versionable.Backends;

/// <summary>
/// Options passed through <c>Save</c> to a backend.
/// </summary>
/// <remarks>
/// Python counterpart: the keyword arguments <c>save()</c> forwards to
/// <c>Backend.save()</c> in <c>src/versionable/_api.py</c> — <c>commentDefaults</c> plus
/// <c>**kwargs</c> such as the HDF5 backend's <c>compression</c>. C# keeps the one
/// cross-backend option typed and puts the rest in
/// <see cref="BackendOptions"/>; backends ignore options they do not understand, as Python
/// does.
/// </remarks>
public sealed record BackendSaveOptions
{
    /// <summary>Options with no cross-backend meaning, keyed by backend-defined names.</summary>
    public IReadOnlyDictionary<string, object?> BackendOptions { get; init; } =
        new Dictionary<string, object?>(StringComparer.Ordinal);

    /// <summary>
    /// Write fields still at their default value as commented-out lines, where the format
    /// has comments (TOML, YAML). Ignored elsewhere. Python counterpart:
    /// <c>commentDefaults</c>.
    /// </summary>
    /// <remarks>
    /// Two backends implement this — <c>Versionable.Backends.Yaml.YamlBackend</c> and
    /// <c>Versionable.Backends.Toml.TomlEmitter</c> — and they agree on what it means, which
    /// Python's two do not. The rules, and the three places C# deliberately departs from Python:
    /// <list type="number">
    ///   <item>
    ///     <description>
    ///     <b>A defaulted field is commented out whole, including when it is a block.</b> A nested
    ///     object or a list of them left at its default takes its header, its keys, and its
    ///     <c>__versionable__</c> sub-table into the comment with it. Python's TOML backend leaves
    ///     the header and the envelope live and comments only the data keys under them, which
    ///     writes an envelope declaring an object the file does not actually set; "not set" already
    ///     means "absent" for every other field, and a partly-commented block is not something a
    ///     reader can uncomment correctly.
    ///     </description>
    ///   </item>
    ///   <item>
    ///     <description>
    ///     <b>A commented block carries its full dotted path.</b> TOML writes
    ///     <c># [[inner.items]]</c> where Python — which renders the default block in isolation —
    ///     writes <c># [[items]]</c>. Deleting the <c>#</c> is the entire point of the line, and
    ///     Python's spelling lands the values in a root field of that name rather than in the
    ///     field the default came from.
    ///     </description>
    ///   </item>
    ///   <item>
    ///     <description>
    ///     <b>Blank lines inside a commented block are dropped rather than written as a bare
    ///     <c>#</c>.</b> Python emits the empty comment; nothing reads it either way.
    ///     </description>
    ///   </item>
    /// </list>
    /// <para>
    /// <b>A commented block must survive being uncommented</b>, which is the whole reason to write
    /// one, and in TOML that constrains where it goes: a bare <c>key = value</c> binds to the most
    /// recent table header, so a live key printed after a commented <c>[section]</c> would join
    /// that block's last table the moment a reader deletes the <c>#</c>. TOML therefore emits
    /// commented sections with the other sections, after every key/value line at their level,
    /// rather than inline where the field is declared. YAML needs no such rule: its comments are
    /// indentation-scoped and a commented block is positionally inert.
    /// </para>
    /// <para>
    /// One difference between the two C# backends remains, and it is the formats': TOML sections
    /// are addressable, so a nested object the caller <em>did</em> set keeps its section and the
    /// defaults still standing inside it are commented in place. YAML comments top-level fields
    /// only — a nested mapping is emitted by the block emitter as one unit. Both, and the
    /// uncommenting semantic above, are pinned by <c>YamlTomlCommentDefaultsTests</c>.
    /// </para>
    /// <para>
    /// The envelope itself is never commented in either format, whatever the fields do: without it
    /// the file loads version-less and, through the dynamic entry point, type-less.
    /// </para>
    /// </remarks>
    public bool CommentDefaults { get; init; }

    /// <summary>
    /// Omit fields still at their declared default, overriding the type's own
    /// <see cref="VersionableMetadata.SkipDefaults"/> for this one save.
    /// </summary>
    /// <remarks>
    /// <see langword="null"/>, the default, means "whatever the type declares", which is the only
    /// setting Python has: <c>skip_defaults</c> there is a class parameter, not a <c>save()</c>
    /// argument. So the zero-configuration behavior is Python's exactly, and this is an escape
    /// hatch for a caller who wants a compact file out of a type that did not ask for one — or a
    /// complete file out of a type that did.
    /// <para>
    /// <b>Where it applies.</b> To the root object's fields only, which is where Python applies
    /// <c>skip_defaults</c> (in <c>save()</c>, before the backend is called). A nested object
    /// always writes every field: a nested value that silently dropped fields would be
    /// indistinguishable from one whose fields were never written, and the enclosing file gives a
    /// reader nothing to reconstruct them from.
    /// </para>
    /// <para>
    /// <b>What counts as "at its default" — and the limit on it.</b> The comparison is structural
    /// and happens on the wire, so an empty <c>List&lt;int&gt;</c> counts as equal to an empty
    /// declared default and a nested object counts as equal to an equal nested default; the
    /// question asked is whether the two would write the same file. But it can only compare
    /// against a default the generator
    /// could see: <see cref="FieldDescriptor.HasDefault"/> covers <b>literal initializers and
    /// closure-free empty constructions</b> — <c>= 8080</c>, <c>= "anon"</c>, <c>= null</c>,
    /// <c>= default</c>, <c>= new()</c>, <c>= []</c>. Everything else is invisible: a default
    /// assigned in a constructor body, an initializer that closes over other state, an object
    /// initializer such as <c>= new Inner { Depth = 2 }</c>, and a property with no initializer at
    /// all. A field whose default is invisible is never skipped and is always written. That is the
    /// safe direction — a field that could have been omitted merely appears — and it is the
    /// deliberate limit of the feature rather than a bug to widen the generator for.
    /// </para>
    /// <para>
    /// Loads round-trip either way: an omitted field is absent from the file, and an absent field
    /// materializes from the same declared default it was omitted for.
    /// </para>
    /// </remarks>
    public bool? SkipDefaults { get; init; }
}
