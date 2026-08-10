namespace Versionable;

/// <summary>
/// Implemented by the source generator on every <c>[Versionable]</c> type, exposing that
/// type's metadata as a static member.
/// </summary>
/// <remarks>
/// <para>
/// <b>The declaring type must be <c>partial</c>.</b> The generator implements this interface
/// by emitting a second part of the type, and a static abstract member cannot be implemented
/// from outside the type that declares it. A non-partial <c>[Versionable]</c> type therefore
/// gets no metadata member and drops out of the compile-time path entirely. Declare every
/// <c>[Versionable]</c> type <c>partial</c>. The analyzer diagnoses the omission as
/// <c>VSN0005</c> at build time.
/// </para>
/// <para>
/// This is the reflection-free path required by ADR-0003: where the type is known at compile
/// time, <c>T.VersionableMetadata</c> reaches the metadata with no registry lookup and no
/// reflection, which is what keeps the engine Native AOT and trimming safe. Reaching it takes
/// a generic constraint — a static abstract member is only accessible through a type
/// parameter, so any API that wants it must be written
/// <c>Method&lt;T&gt;(...) where T : IVersionableMetadataProvider</c>.
/// </para>
/// <para>
/// Python counterpart: the <c>_serializer_meta_</c> class attribute set by
/// <c>Versionable.__init_subclass__</c> (<c>src/versionable/_base.py</c>) and read back by
/// <c>metadata(cls)</c>. Like <c>_serializer_meta_</c>, it is present regardless of
/// <c>Register</c> — which is why a <c>Register = false</c> type is still serializable.
/// </para>
/// <para>
/// <b>Identity contract.</b> <see cref="VersionableMetadata"/> returns the same instance on
/// every call, and it is the same instance the generated <c>[ModuleInitializer]</c> hands to
/// <see cref="VersionableRegistry.Register"/> or
/// <see cref="VersionableRegistry.RegisterTypeOnly"/>. For a registered type,
/// <c>ReferenceEquals(T.VersionableMetadata, registryResult)</c> therefore holds: the registry
/// stores the instance it was given and never copies it. Engine code may cache either one
/// without tracking which it captured, and reference equality is a valid identity check.
/// </para>
/// <para>
/// The generator emits the implementation; users never write it. Code that only has a
/// <see cref="Type"/> at runtime — polymorphic saves, for instance — goes through
/// <see cref="VersionableRegistry.TryGetByType"/> instead.
/// </para>
/// </remarks>
public interface IVersionableMetadataProvider
{
    /// <summary>
    /// Metadata describing this type's schema, fields, and accessors. The same instance on
    /// every call, and the same instance held by <see cref="VersionableRegistry"/>.
    /// </summary>
    static abstract VersionableMetadata VersionableMetadata { get; }
}
