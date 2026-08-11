namespace Versionable.Backends;

/// <summary>
/// What <see cref="IVersionableBackend.Load"/> returns.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>(fields_dict, metadata_dict)</c> tuple returned by
/// <c>Backend.load()</c> in <c>src/versionable/_backend.py</c>, plus the third element
/// (<c>lazy_fields</c>) that <c>loadLazy()</c> adds. Named here because tuple element names do
/// not survive an interface boundary readably.
/// </remarks>
/// <param name="Fields">
/// Raw values keyed by the wire names found in the file, envelope keys already stripped (see
/// <see cref="VersionableEnvelope.ReservedKeys"/>). Migrations run over this dictionary, so it
/// holds file-shaped keys, which need not be the target type's fields.
/// </param>
/// <param name="Envelope">The envelope read from the file, in either wire layout.</param>
/// <param name="LazyFields">
/// Wire names present in the file but deliberately not materialized in
/// <paramref name="Fields"/>, because <see cref="BackendLoadOptions.Preload"/> or
/// <see cref="BackendLoadOptions.MetadataOnly"/> excluded them. Empty for backends that always
/// load eagerly.
/// <para>
/// <b>Entries are <c>/</c>-separated chains of <em>field names</em> relative to the load root</b>:
/// <c>values</c> for a field of the root object, <c>inner/values</c> for a field of a nested
/// object. A flat name is simply the zero-depth case, so a backend whose skips are all root-level
/// — and a schema with no nesting — needs to do nothing differently.
/// </para>
/// <para>
/// <b>Container indexes and keys never appear.</b> A <c>Dictionary&lt;string, List&lt;Inner&gt;&gt;</c>
/// named <c>groups</c> whose element type skips <c>samples</c> records <c>groups/samples</c> once,
/// however many elements there are and however deeply the containers nest. Every element of a
/// container has one declared type, so a field skipped in one is skipped in all, and an index is
/// not part of a skip's identity. It is also not reconstructable: the generated reader that
/// materializes a container hands the engine an element, never which element, so a path carrying
/// <c>0</c> could never be matched back to anything.
/// </para>
/// <para>
/// The qualification is what lets the engine tell <em>which</em> object skipped a field. Without
/// it, a nested skip is indistinguishable from a field the file never had, and the load reports
/// a missing field instead of a skipped one — which is the bug this convention exists to make
/// unrepresentable.
/// </para>
/// </param>
public sealed record BackendLoadResult(
    IReadOnlyDictionary<string, object?> Fields,
    EnvelopeMetadata Envelope,
    IReadOnlySet<string>? LazyFields = null);
