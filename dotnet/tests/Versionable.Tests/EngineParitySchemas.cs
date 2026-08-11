using Versionable.Errors;
using Versionable.Migrations;

namespace Versionable.Tests;

/// <summary>
/// Fixtures for the polymorphism and <c>SkipDefaults</c> parity suites: a small class hierarchy
/// with one of every resolution outcome, and a type whose defaults only a structural comparison
/// can recognise.
/// </summary>
/// <remarks>
/// Python counterpart: the throwaway classes defined inside <c>TestPolymorphism</c> and
/// <c>test_polymorphism_*</c> in <c>tests/test_nested_migrations.py</c>. Python declares them per
/// test because a class there is a runtime value; C# types are compile-time, so the whole cast is
/// declared once and each test picks the members it needs.
/// <para>
/// Real <c>[Versionable]</c> types rather than hand-written metadata, because the generated
/// <c>WireReader</c> for a <c>List&lt;Base&gt;</c> and a <c>Dictionary&lt;string, Base&gt;</c> is
/// half of what these tests are checking — a reader that named the concrete type instead of
/// handing the element to <see cref="Versionable.Engine.WireValues.ReadVersionable"/> would make
/// every polymorphic load quietly return the base type.
/// </para>
/// </remarks>
internal static class ParitySchemas
{
    /// <summary>Every fixture in this file, for the registry reset seam.</summary>
    internal static IReadOnlyList<VersionableMetadata> All { get; } =
    [
        PolyAnimal.VersionableMetadata,
        PolyDog.VersionableMetadata,
        PolyCat.VersionableMetadata,
        PolyWolf.VersionableMetadata,
        PolyOwl.VersionableMetadata,
        PolyPack.VersionableMetadata,
        PolyVehicle.VersionableMetadata,
        PolyZoo.VersionableMetadata,
        EngineCompactInner.VersionableMetadata,
        EngineCompact.VersionableMetadata,
    ];

    /// <summary>The fixtures that claim no Serialization Name, for the same seam.</summary>
    internal static IReadOnlyList<VersionableMetadata> Unregistered { get; } =
    [
        PolyGhost.VersionableMetadata,
    ];
}

/// <summary>The polymorphic base. Concrete, so it is loadable in its own right.</summary>
[Versionable(Version = 1, Hash = "332689")]
internal partial class PolyAnimal
{
    /// <summary>The animal's name.</summary>
    [VersionableField("name")]
    public required string Name { get; init; }
}

/// <summary>A subclass, itself further subclassable.</summary>
[Versionable(Version = 1, Hash = "f13e36")]
internal partial class PolyDog : PolyAnimal
{
    /// <summary>Breed.</summary>
    [VersionableField("breed")]
    public string Breed { get; init; } = "mutt";
}

/// <summary>Another subclass.</summary>
[Versionable(Version = 1, Hash = "8b138f")]
internal sealed partial class PolyCat : PolyAnimal
{
    /// <summary>Whether the cat lives indoors.</summary>
    [VersionableField("indoor")]
    public bool Indoor { get; init; } = true;
}

/// <summary>A subclass that still answers to the name it was written under.</summary>
[Versionable(Version = 1, Hash = "b13805", OldNames = new[] { "PolyOldWolf" })]
internal sealed partial class PolyWolf : PolyAnimal
{
    /// <summary>Pack size.</summary>
    [VersionableField("pack")]
    public int Pack { get; init; }
}

/// <summary>
/// A subclass at version 2 whose version-1 files spell <c>sound</c> as <c>call</c>.
/// </summary>
/// <remarks>
/// The point of the fixture is that the migration belongs to the <em>element's</em> type: a
/// <c>List&lt;PolyAnimal&gt;</c> holding one of these has to run this chain and not the base's
/// (which has none), which only happens if the nested load resolves the concrete type before it
/// decides what to migrate.
/// </remarks>
[Versionable(Version = 2, Hash = "5b9ce8")]
internal sealed partial class PolyOwl : PolyAnimal
{
    /// <summary>The noise it makes; <c>call</c> before the v1 migration renamed it.</summary>
    [VersionableField("sound")]
    public string Sound { get; init; } = "hoot";

    /// <summary>The migration chain, in the hand-written form.</summary>
    internal sealed class Migrate : IMigrationChain
    {
        /// <inheritdoc/>
        public IReadOnlyList<int> FromVersions { get; } = [1];

        /// <inheritdoc/>
        public int? MinReversibleVersion => null;

        /// <inheritdoc/>
        public IDictionary<string, object?> Apply(
            IDictionary<string, object?> fields,
            int fromVersion,
            int toVersion,
            bool upgradeInPlace)
        {
            ArgumentNullException.ThrowIfNull(fields);

            for (int version = fromVersion; version < toVersion; version++)
            {
                if (version != 1)
                {
                    throw new MigrationException($"PolyOwl has no migration from version {version}.");
                }

                if (fields.Remove("call", out object? call))
                {
                    fields["sound"] = call;
                }
            }

            return fields;
        }
    }
}

/// <summary>A subclass that itself holds a polymorphic collection.</summary>
[Versionable(Version = 1, Hash = "976313")]
internal sealed partial class PolyPack : PolyAnimal
{
    /// <summary>The members, each resolved from its own envelope.</summary>
    [VersionableField("members")]
    public List<PolyAnimal> Members { get; init; } = [];
}

/// <summary>A subclass that claims no Serialization Name, so no file can name it.</summary>
[Versionable(Version = 1, Hash = "aacf13", Register = false)]
internal sealed partial class PolyGhost : PolyAnimal
{
    /// <summary>Where it is seen.</summary>
    [VersionableField("haunts")]
    public string Haunts { get; init; } = "";
}

/// <summary>Registered, and not an animal — the wrong-branch case.</summary>
[Versionable(Version = 1, Hash = "426ffe")]
internal sealed partial class PolyVehicle
{
    /// <summary>Wheel count.</summary>
    [VersionableField("wheels")]
    public required int Wheels { get; init; }
}

/// <summary>Collections declared as the base type, in all three shapes a field can take.</summary>
[Versionable(Version = 1, Hash = "6c1fcb")]
internal sealed partial class PolyZoo
{
    /// <summary>A list of the base type.</summary>
    [VersionableField("animals")]
    public List<PolyAnimal> Animals { get; init; } = [];

    /// <summary>A dictionary valued by the base type.</summary>
    [VersionableField("byName")]
    public Dictionary<string, PolyAnimal> ByName { get; init; } = new();

    /// <summary>A single, optional value of the base type.</summary>
    /// <remarks>
    /// The declared default matters beyond the field itself: TOML has no null literal and omits
    /// the key, so a null <c>star</c> is only round-trippable because there is a default to
    /// rebuild it from.
    /// </remarks>
    [VersionableField("star")]
    public PolyAnimal? Star { get; init; } = null;
}

/// <summary>A nested object whose own fields have defaults.</summary>
/// <remarks>
/// Two of them, so a test can leave one at its default while making the object as a whole differ
/// from the enclosing field's default — which is how "a nested object writes every field" is
/// distinguishable from "this nested object happened to have nothing to skip".
/// </remarks>
[Versionable(Version = 1, Hash = "1a5a44")]
internal sealed partial class EngineCompactInner
{
    /// <summary>How deep.</summary>
    [VersionableField("depth")]
    public int Depth { get; init; } = 1;

    /// <summary>What to call it.</summary>
    [VersionableField("label")]
    public string Label { get; init; } = "root";
}

/// <summary>
/// A type declared <c>SkipDefaults</c> whose defaults are mostly containers and objects.
/// </summary>
/// <remarks>
/// The scalar case works under any comparison; an empty <c>List&lt;string&gt;</c> and an
/// <c>EngineCompactInner</c> equal to the declared one do not, because neither is
/// <see cref="object.Equals(object?, object?)"/>-equal to a freshly built default. They are what
/// the wire comparison exists for.
/// </remarks>
[Versionable(Version = 1, Hash = "c46573", SkipDefaults = true)]
internal sealed partial class EngineCompact
{
    /// <summary>A scalar default.</summary>
    [VersionableField("name")]
    public string Name { get; init; } = "anon";

    /// <summary>A list default; reference equality would never see it.</summary>
    [VersionableField("tags")]
    public List<string> Tags { get; init; } = [];

    /// <summary>A dictionary default.</summary>
    [VersionableField("limits")]
    public Dictionary<string, int> Limits { get; init; } = new();

    /// <summary>A nested-object default.</summary>
    [VersionableField("inner")]
    public EngineCompactInner Inner { get; init; } = new();

    /// <summary>A field with no default, which is therefore never skipped.</summary>
    [VersionableField("required")]
    public required int Required { get; init; }
}
