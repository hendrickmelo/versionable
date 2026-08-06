using System.Text.Json;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Collections declared as a base type: what save writes for each element, what load resolves it
/// back to, and every way that resolution can fail.
/// </summary>
/// <remarks>
/// Python counterpart: <c>TestPolymorphism</c>, <c>test_polymorphism_with_migration</c>, and
/// <c>test_polymorphism_old_names_rename</c> in <c>tests/test_nested_migrations.py</c>. The golden
/// corpus already proves a <c>List&lt;Base&gt;</c> round-trips on all four backends; these are the
/// cases a fixture cannot express — a name the registry does not know, a name that resolves to the
/// wrong branch of the type graph, and an element whose own migration chain has to run.
/// <para>
/// Error cases are hand-authored JSON, as Python's are: they describe files no writer produces,
/// so there is nothing to save first.
/// </para>
/// </remarks>
[Collection(RegistryCollection.Name)]
public class EnginePolymorphismTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"versionable-poly-{Path.GetRandomFileName()}");

    public EnginePolymorphismTests()
    {
        GoldenSchemas.EnsureRegistered();
        Directory.CreateDirectory(_directory);
    }

    public void Dispose()
    {
        Directory.Delete(_directory, recursive: true);
        GC.SuppressFinalize(this);
    }

    // ------------------------------------------------------------------
    // Round trip
    // ------------------------------------------------------------------

    [Theory]
    [InlineData(".json")]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    [InlineData(".h5")]
    public void every_base_typed_field_comes_back_as_the_type_it_was_saved_as(string extension)
    {
        PolyZoo zoo = new()
        {
            Animals =
            [
                new PolyDog { Name = "Rex", Breed = "lab" },
                new PolyCat { Name = "Whiskers", Indoor = false },

                // The base is concrete, so an element may legitimately be one.
                new PolyAnimal { Name = "Generic" },
            ],
            ByName = new Dictionary<string, PolyAnimal>(StringComparer.Ordinal)
            {
                ["pet"] = new PolyCat { Name = "Tom", Indoor = true },
                ["wild"] = new PolyWolf { Name = "Fang", Pack = 7 },
            },
            Star = new PolyDog { Name = "Lassie", Breed = "collie" },
        };

        PolyZoo loaded = RoundTrip(zoo, extension);

        Assert.Equal("lab", Assert.IsType<PolyDog>(loaded.Animals[0]).Breed);
        Assert.False(Assert.IsType<PolyCat>(loaded.Animals[1]).Indoor);
        Assert.Equal("Generic", Assert.IsType<PolyAnimal>(loaded.Animals[2]).Name);
        Assert.True(Assert.IsType<PolyCat>(loaded.ByName["pet"]).Indoor);
        Assert.Equal(7, Assert.IsType<PolyWolf>(loaded.ByName["wild"]).Pack);
        Assert.Equal("collie", Assert.IsType<PolyDog>(loaded.Star).Breed);
    }

    [Theory]
    [InlineData(".json")]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    [InlineData(".h5")]
    public void a_polymorphic_element_can_itself_hold_a_polymorphic_collection(string extension)
    {
        PolyZoo zoo = new()
        {
            Animals =
            [
                new PolyPack
                {
                    Name = "Alpha",
                    Members = [new PolyWolf { Name = "Fang", Pack = 2 }, new PolyCat { Name = "Interloper" }],
                },
            ],
        };

        PolyZoo loaded = RoundTrip(zoo, extension);
        PolyPack pack = Assert.IsType<PolyPack>(loaded.Animals[0]);

        Assert.Equal(2, Assert.IsType<PolyWolf>(pack.Members[0]).Pack);
        Assert.IsType<PolyCat>(pack.Members[1]);

        // TOML has no null literal and drops the key entirely, so an unset `star` only survives
        // the trip because the field declares a default to rebuild it from.
        Assert.Null(loaded.Star);
    }

    [Fact]
    public void each_element_carries_its_own_envelope_rather_than_the_declared_types()
    {
        // The whole mechanism rests on this: a reader has nothing but the element's envelope to
        // resolve it by, so a writer that stamped the declared type's name on every element would
        // round-trip a Dog into an Animal and lose `breed` on the way.
        string path = Path.Combine(_directory, "zoo.json");
        VersionableFile.Save(
            new PolyZoo { Animals = [new PolyDog { Name = "Rex" }, new PolyCat { Name = "Tom" }] },
            path);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement animals = document.RootElement.GetProperty("animals");

        Assert.Equal("PolyZoo", Envelope(document.RootElement));
        Assert.Equal("PolyDog", Envelope(animals[0]));
        Assert.Equal("PolyCat", Envelope(animals[1]));
    }

    // ------------------------------------------------------------------
    // Resolution
    // ------------------------------------------------------------------

    [Fact]
    public void an_element_naming_a_type_the_registry_does_not_know_is_refused()
    {
        string path = Hand("stranger.json", """{"object": "NotARealType", "version": 1}""", "\"name\": \"Rex\"");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<PolyZoo>(path));

        Assert.Contains("NotARealType", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_element_naming_a_registered_type_off_the_declared_branch_is_refused()
    {
        // PolyVehicle is registered and loadable — it just is not a PolyAnimal, and a list
        // declared `List<PolyAnimal>` cannot hold one whatever the file says.
        string path = Hand("wrong-branch.json", """{"object": "PolyVehicle", "version": 1}""", "\"wheels\": 4");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<PolyZoo>(path));

        Assert.Contains("not assignable", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_element_naming_a_type_that_claims_no_name_is_refused()
    {
        // Register = false keeps the type out of the name index, so it saves — the writer reaches
        // its metadata by CLR type — and then cannot be read back. Worth pinning: the failure is on
        // the load, and the message has to point at the name rather than at the file.
        string path = Path.Combine(_directory, "ghost.json");
        VersionableFile.Save(new PolyZoo { Animals = [new PolyGhost { Name = "Boo", Haunts = "attic" }] }, path);

        Assert.Contains("\"object\": \"PolyGhost\"", File.ReadAllText(path), StringComparison.Ordinal);

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<PolyZoo>(path));

        Assert.Contains("PolyGhost", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_element_naming_a_type_by_a_name_it_used_to_have_resolves_to_it()
    {
        string path = Hand(
            "renamed.json",
            """{"object": "PolyOldWolf", "version": 1}""",
            "\"name\": \"Fang\", \"pack\": 3");

        PolyZoo loaded = VersionableFile.Load<PolyZoo>(path);

        Assert.Equal(3, Assert.IsType<PolyWolf>(loaded.Animals[0]).Pack);
    }

    [Fact]
    public void an_element_with_no_envelope_falls_back_to_the_declared_type()
    {
        // 0.1.x wrote nested objects without one. The declared type is the only answer available,
        // and it is the right one for every file written before subclasses were in play.
        string path = Path.Combine(_directory, "bare.json");
        File.WriteAllText(
            path,
            """
            {
              "__versionable__": {"object": "PolyZoo", "version": 1, "hash": ""},
              "animals": [{"name": "Nameless"}],
              "star": null
            }
            """);

        PolyZoo loaded = VersionableFile.Load<PolyZoo>(path);

        Assert.Equal("Nameless", Assert.IsType<PolyAnimal>(loaded.Animals[0]).Name);
    }

    // ------------------------------------------------------------------
    // Migration
    // ------------------------------------------------------------------

    [Fact]
    public void each_element_is_migrated_by_the_chain_of_the_type_it_actually_is()
    {
        // PolyOwl is at version 2 and renames `call` to `sound`; PolyAnimal, the declared element
        // type, has no chain at all. Resolving the concrete type before deciding what to migrate is
        // the only order in which this file loads.
        string path = Hand(
            "old-owl.json",
            """{"object": "PolyOwl", "version": 1}""",
            "\"name\": \"Hedwig\", \"call\": \"HOOT\"");

        PolyZoo loaded = VersionableFile.Load<PolyZoo>(path);

        PolyOwl owl = Assert.IsType<PolyOwl>(loaded.Animals[0]);
        Assert.Equal("HOOT", owl.Sound);
        Assert.Equal("Hedwig", owl.Name);
    }

    [Fact]
    public void an_element_newer_than_the_type_that_would_load_it_is_refused()
    {
        string path = Hand("future-owl.json", """{"object": "PolyOwl", "version": 9}""", "\"name\": \"Hedwig\"");

        VersionException error = Assert.Throws<VersionException>(() => VersionableFile.Load<PolyZoo>(path));

        Assert.Contains("nested PolyOwl", error.Message, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static string Envelope(JsonElement element) =>
        element.GetProperty(VersionableEnvelope.WrappedKey).GetProperty("object").GetString()!;

    private PolyZoo RoundTrip(PolyZoo zoo, string extension)
    {
        string path = Path.Combine(_directory, $"zoo{extension}");
        VersionableFile.Save(zoo, path);
        return VersionableFile.Load<PolyZoo>(path);
    }

    /// <summary>Writes a zoo whose single element carries <paramref name="envelope"/>.</summary>
    private string Hand(string name, string envelope, string fields)
    {
        string path = Path.Combine(_directory, name);
        File.WriteAllText(
            path,
            $$"""
            {
              "__versionable__": {"object": "PolyZoo", "version": 1, "hash": ""},
              "animals": [{"__versionable__": {{envelope}}, {{fields}}}]
            }
            """);

        return path;
    }
}
