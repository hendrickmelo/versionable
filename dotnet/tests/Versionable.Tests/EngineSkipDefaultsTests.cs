using System.Text.Json;
using Versionable.Backends;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// <c>SkipDefaults</c>: which fields a save omits, and what a load does with the gap.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>skip_defaults</c> loop at the top of <c>save()</c> in
/// <c>src/versionable/_api.py</c>, and <c>test_skip_defaults</c> in <c>tests/test_base.py</c>.
/// <para>
/// The interesting half is not that a defaulted scalar disappears — it is that a defaulted
/// <em>container</em> and a defaulted nested <em>object</em> disappear too. Python gets that free
/// from structural <c>==</c>; C# only gets it by comparing what the two would write, so these
/// tests are what stands between the option and being a no-op for every type that defaults a list.
/// </para>
/// </remarks>
[Collection(RegistryCollection.Name)]
public class EngineSkipDefaultsTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"versionable-skip-{Path.GetRandomFileName()}");

    public EngineSkipDefaultsTests()
    {
        GoldenSchemas.EnsureRegistered();
        Directory.CreateDirectory(_directory);
    }

    public void Dispose()
    {
        Directory.Delete(_directory, recursive: true);
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void a_container_or_object_default_is_recognised_and_not_only_a_scalar_one()
    {
        // None of `tags`, `limits`, or `inner` is Equals-equal to a freshly built default, so an
        // identity comparison would write all three and leave the option doing nothing useful.
        Assert.Equal(["required"], Keys(Save(new EngineCompact { Required = 3 }, "all-default.json")));
    }

    [Fact]
    public void a_field_that_differs_from_its_default_is_written()
    {
        EngineCompact compact = new()
        {
            Required = 3,
            Tags = ["alpha"],
            Limits = new Dictionary<string, int>(StringComparer.Ordinal) { ["max"] = 9 },
            Inner = new EngineCompactInner { Depth = 5 },
        };

        Assert.Equal(["inner", "limits", "required", "tags"], Keys(Save(compact, "changed.json")));
    }

    [Fact]
    public void a_field_with_no_visible_default_is_always_written()
    {
        // `required` has no initializer, so the generator records none and there is nothing for it
        // to be equal to. Same for every default the generator cannot see — see BackendSaveOptions.
        Assert.Contains("required", Keys(Save(new EngineCompact { Required = 0 }, "zeroed.json")));
    }

    [Fact]
    public void a_nested_object_writes_every_field_even_when_the_root_skips()
    {
        // Python applies skip_defaults in save(), to the root's fields, and never below. A nested
        // object that dropped its own defaulted fields would be indistinguishable from one whose
        // fields were never written, and the file would give a reader nothing to tell them apart.
        JsonElement root = Save(
            new EngineCompact { Required = 3, Inner = new EngineCompactInner { Depth = 5 } },
            "nested.json");

        JsonElement inner = root.GetProperty("inner");

        Assert.Equal(5, inner.GetProperty("depth").GetInt32());
        Assert.Equal("root", inner.GetProperty("label").GetString());
    }

    [Theory]
    [InlineData(".json")]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    [InlineData(".h5")]
    public void the_skipped_keys_are_absent_from_what_the_backend_reads_back(string extension)
    {
        // Python counterpart: the h5py assertions in test_hdf5_backend.py's TestSkipDefaults —
        // `"count" not in f.attrs`. The round-trip theory below cannot stand in for this: it
        // passes just as well when nothing is skipped at all, because an omitted field and a
        // written-at-its-default field materialize identically. Reading back through the backend
        // rather than through the file format is what makes one assertion serve all four.
        string path = Path.Combine(_directory, $"absent{extension}");
        VersionableFile.Save(new EngineCompact { Required = 3 }, path);

        BackendLoadResult result = BackendRegistry.Resolve(path).Load(
            path,
            new BackendLoadOptions { TargetMetadata = EngineCompact.VersionableMetadata });

        Assert.Equal(["required"], result.Fields.Keys.Order(StringComparer.Ordinal));
    }

    [Theory]
    [InlineData(".json")]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    [InlineData(".h5")]
    public void an_omitted_field_comes_back_from_the_default_it_was_omitted_for(string extension)
    {
        string path = Path.Combine(_directory, $"compact{extension}");
        VersionableFile.Save(new EngineCompact { Required = 3 }, path);

        EngineCompact loaded = VersionableFile.Load<EngineCompact>(path);

        Assert.Equal("anon", loaded.Name);
        Assert.Empty(loaded.Tags);
        Assert.Empty(loaded.Limits);
        Assert.Equal(1, loaded.Inner.Depth);
        Assert.Equal("root", loaded.Inner.Label);
        Assert.Equal(3, loaded.Required);
    }

    // ------------------------------------------------------------------
    // The save-time override
    // ------------------------------------------------------------------

    [Fact]
    public void the_save_option_can_turn_a_types_own_declaration_off()
    {
        JsonElement written = Save(
            new EngineCompact { Required = 3 },
            "verbose.json",
            new BackendSaveOptions { SkipDefaults = false });

        Assert.Equal(["inner", "limits", "name", "required", "tags"], Keys(written));
    }

    [Fact]
    public void the_save_option_can_turn_skipping_on_for_a_type_that_did_not_ask()
    {
        string path = Path.Combine(_directory, "settings.json");
        VersionableFile.Save(
            new EngineSettings("anon", 8080, "needed"),
            path,
            options: new BackendSaveOptions { SkipDefaults = true });

        Assert.Equal(["required"], Keys(Parse(path)));
        Assert.Equal("anon", VersionableFile.Load<EngineSettings>(path).Name);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static IReadOnlyList<string> Keys(JsonElement root) =>
        [.. root.EnumerateObject()
            .Select(property => property.Name)
            .Where(name => !string.Equals(name, VersionableEnvelope.WrappedKey, StringComparison.Ordinal))
            .Order(StringComparer.Ordinal)];

    private static JsonElement Parse(string path) =>
        JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone();

    private JsonElement Save(EngineCompact compact, string name, BackendSaveOptions? options = null)
    {
        string path = Path.Combine(_directory, name);
        VersionableFile.Save(compact, path, options: options);
        return Parse(path);
    }
}
