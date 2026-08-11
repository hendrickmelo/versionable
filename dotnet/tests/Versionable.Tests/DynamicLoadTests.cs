using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// <see cref="VersionableFile.LoadDynamic(string, Versionable.Backends.IVersionableBackend?,
/// VersionableLoadOptions?)"/>: loading a file whose type the caller does not name.
/// </summary>
/// <remarks>
/// Python counterpart: <c>loadDynamic()</c> in <c>src/versionable/_api.py</c>. The name is
/// resolved from the envelope through the registry and everything after that is the ordinary
/// load, so what needs pinning is the resolution — which names resolve, which are refused, and
/// that nothing downstream of the resolution is skipped.
/// </remarks>
[Collection(RegistryCollection.Name)]
public class DynamicLoadTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"versionable-dynamic-{Path.GetRandomFileName()}");

    public DynamicLoadTests()
    {
        GoldenSchemas.EnsureRegistered();
        Directory.CreateDirectory(_directory);
    }

    public void Dispose()
    {
        Directory.Delete(_directory, recursive: true);
        GC.SuppressFinalize(this);
    }

    [Theory]
    [InlineData(".json")]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    [InlineData(".h5")]
    public void the_type_and_its_values_come_back_from_the_envelope_on_every_backend(string extension)
    {
        string path = Path.Combine(_directory, $"dog{extension}");
        VersionableFile.Save(new PolyDog { Name = "Rex", Breed = "lab" }, path);

        PolyDog loaded = Assert.IsType<PolyDog>(VersionableFile.LoadDynamic(path));

        Assert.Equal("Rex", loaded.Name);
        Assert.Equal("lab", loaded.Breed);
    }

    [Fact]
    public void a_bound_accepts_a_subclass_of_itself()
    {
        // Python counterpart: loadDynamic(path, baseClass=Animal).
        string path = Path.Combine(_directory, "wolf.json");
        VersionableFile.Save(new PolyWolf { Name = "Fang", Pack = 4 }, path);

        PolyAnimal loaded = VersionableFile.LoadDynamic<PolyAnimal>(path);

        Assert.Equal(4, Assert.IsType<PolyWolf>(loaded).Pack);
    }

    [Fact]
    public void a_bound_refuses_a_type_off_its_own_branch()
    {
        string path = Path.Combine(_directory, "vehicle.json");
        VersionableFile.Save(new PolyVehicle { Wheels = 4 }, path);

        BackendException error =
            Assert.Throws<BackendException>(() => VersionableFile.LoadDynamic<PolyAnimal>(path));

        Assert.Contains("not assignable", error.Message, StringComparison.Ordinal);
        Assert.Contains("PolyVehicle", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_name_the_registry_does_not_know_is_refused()
    {
        string path = Write("stranger.json", """{"object": "NotRegistered", "version": 1, "hash": "000000"}""");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.LoadDynamic(path));

        Assert.Contains("Unknown object type 'NotRegistered'", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_file_naming_no_type_is_refused_with_the_typed_entry_point_named()
    {
        // Nothing can be resolved from a version-only envelope, and the caller's next move is the
        // generic overload — so the message says so rather than reporting a corrupt file.
        string path = Write("anonymous.json", """{"version": 1}""");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.LoadDynamic(path));

        Assert.Contains("records no object name", error.Message, StringComparison.Ordinal);
        Assert.Contains("Load<T>()", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_name_the_type_used_to_have_resolves_to_it()
    {
        string path = Write(
            "renamed.json",
            """{"object": "PolyOldWolf", "version": 1, "hash": ""}""",
            "\"name\": \"Fang\", \"pack\": 2");

        Assert.Equal(2, Assert.IsType<PolyWolf>(VersionableFile.LoadDynamic(path)).Pack);
    }

    [Fact]
    public void a_type_that_claims_no_name_cannot_be_reached_dynamically()
    {
        // Register = false saves — the writer reaches metadata by CLR type — and is unreadable
        // without a target type, which is the whole meaning of the flag.
        string path = Path.Combine(_directory, "ghost.json");
        VersionableFile.Save(new PolyGhost { Name = "Boo", Haunts = "attic" }, path);

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.LoadDynamic(path));

        Assert.Contains("PolyGhost", error.Message, StringComparison.Ordinal);
        Assert.Equal("attic", VersionableFile.Load<PolyGhost>(path).Haunts);
    }

    [Fact]
    public void a_dynamic_load_still_runs_the_migrations_the_file_needs()
    {
        // The resolution is the only thing that differs from Load<T>(); everything after it —
        // version dispatch, the chain, defaults — has to happen exactly as it otherwise would.
        string path = Write(
            "old-owl.json",
            """{"object": "PolyOwl", "version": 1, "hash": ""}""",
            "\"name\": \"Hedwig\", \"call\": \"HOOT\"");

        Assert.Equal("HOOT", Assert.IsType<PolyOwl>(VersionableFile.LoadDynamic(path)).Sound);
    }

    [Fact]
    public void load_options_reach_a_dynamic_load()
    {
        string path = Write(
            "version-less-owl.json",
            """{"object": "PolyOwl", "hash": ""}""",
            "\"name\": \"Hedwig\", \"call\": \"HOOT\"");

        PolyOwl loaded = Assert.IsType<PolyOwl>(
            VersionableFile.LoadDynamic(path, options: new VersionableLoadOptions { AssumeVersion = 1 }));

        Assert.Equal("HOOT", loaded.Sound);
    }

    private string Write(string name, string envelope, string? fields = null)
    {
        string path = Path.Combine(_directory, name);
        string body = fields is null ? string.Empty : $", {fields}";
        File.WriteAllText(path, $$"""{"__versionable__": {{envelope}}{{body}}}""");
        return path;
    }
}
