namespace Versionable;

/// <summary>
/// The envelope of one serialized object, read from either wire layout.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>meta</c> dict passed to <c>Backend.save()</c>
/// (<c>{"name", "version", "hash"}</c>, built in <c>save()</c> in
/// <c>src/versionable/_api.py</c>) and the dict returned by <c>_readNestedEnvelope</c>
/// (<c>{"object", "version", "hash"}</c>) in <c>src/versionable/_types.py</c>. All members
/// are nullable because 0.1.x files and hand-written files may omit any of them; a missing
/// version falls back to <c>assumeVersion</c> at load.
/// </remarks>
/// <param name="ObjectName">Serialization Name of the object, or <see langword="null"/> when absent.</param>
/// <param name="Version">Schema version the file was written at, or <see langword="null"/> when absent.</param>
/// <param name="Hash">Schema hash the file was written at, or <see langword="null"/> when absent.</param>
public sealed record EnvelopeMetadata(string? ObjectName, int? Version, string? Hash);
