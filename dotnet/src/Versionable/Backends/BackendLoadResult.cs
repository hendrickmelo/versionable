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
/// </param>
public sealed record BackendLoadResult(
    IReadOnlyDictionary<string, object?> Fields,
    EnvelopeMetadata Envelope,
    IReadOnlySet<string>? LazyFields = null);
