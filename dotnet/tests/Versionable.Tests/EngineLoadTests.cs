using System.Collections.Frozen;
using Versionable.Backends;
using Versionable.Engine;
using Versionable.Errors;
using Versionable.Migrations;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The load path: version dispatch, migrations, unknown fields, defaults, literals, and the
/// option translation the backend contract's inverted default makes necessary.
/// </summary>
[Collection(RegistryCollection.Name)]
public class EngineLoadTests
{
    public EngineLoadTests() => GoldenSchemas.EnsureRegistered();

    // ------------------------------------------------------------------
    // Version dispatch
    // ------------------------------------------------------------------

    [Fact]
    public void a_file_at_the_current_version_loads_without_migrating()
    {
        EngineRenameChain chain = new([1, 2], null);
        VersionableMetadata metadata = EngineVersioned.At(chain);

        EngineVersioned loaded = Load<EngineVersioned>(metadata, new() { ["name_v3"] = "current" }, version: 3);

        Assert.Equal("current", loaded.Name);
        Assert.Empty(chain.Applied);
    }

    [Fact]
    public void an_older_file_runs_every_migration_step_in_order()
    {
        EngineRenameChain chain = new([1, 2], null);
        VersionableMetadata metadata = EngineVersioned.At(chain);

        EngineVersioned loaded = Load<EngineVersioned>(metadata, new() { ["name_v1"] = "ancient" }, version: 1);

        Assert.Equal("ancient", loaded.Name);
        Assert.Equal([1, 2], chain.Applied);
    }

    [Fact]
    public void a_newer_file_is_refused_rather_than_downgraded()
    {
        VersionException error = Assert.Throws<VersionException>(
            () => Load<EngineVersioned>(EngineVersioned.Metadata, new() { ["name_v3"] = "future" }, version: 4));

        Assert.Contains("newer than class version", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_older_file_with_no_declared_migrations_is_refused()
    {
        VersionableMetadata metadata = EngineVersioned.At(chain: null);

        VersionException error = Assert.Throws<VersionException>(
            () => Load<EngineVersioned>(metadata, new() { ["name_v1"] = "ancient" }, version: 1));

        Assert.Contains("declares no migrations", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_hole_in_the_chain_is_refused_rather_than_skipped()
    {
        VersionableMetadata metadata = EngineVersioned.At(new EngineRenameChain([1], null));

        VersionException error = Assert.Throws<VersionException>(
            () => Load<EngineVersioned>(metadata, new() { ["name_v1"] = "ancient" }, version: 1));

        Assert.Contains("No migration from version 2 to 3", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_file_older_than_the_chain_reaches_is_refused()
    {
        VersionableMetadata metadata = EngineVersioned.At(new EngineRenameChain([2], minReversibleVersion: 2));

        VersionException error = Assert.Throws<VersionException>(
            () => Load<EngineVersioned>(metadata, new() { ["name_v1"] = "ancient" }, version: 1));

        Assert.Contains("no longer supported", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_file_with_no_version_is_treated_as_current_and_warned_about()
    {
        List<string> warnings = [];
        void Capture(string message) => warnings.Add(message);
        VersionableLog.Warning += Capture;
        try
        {
            EngineVersioned loaded = Load<EngineVersioned>(
                EngineVersioned.Metadata, new() { ["name_v3"] = "guessed" }, version: null);

            Assert.Equal("guessed", loaded.Name);
        }
        finally
        {
            VersionableLog.Warning -= Capture;
        }

        Assert.Contains(warnings, message => message.Contains("No version found", StringComparison.Ordinal));
    }

    [Fact]
    public void assume_version_applies_the_migrations_a_version_less_file_needs()
    {
        EngineRenameChain chain = new([1, 2], null);
        VersionableMetadata metadata = EngineVersioned.At(chain);

        EngineVersioned loaded = Load<EngineVersioned>(
            metadata,
            new() { ["name_v1"] = "ancient" },
            version: null,
            options: new VersionableLoadOptions { AssumeVersion = 1 });

        Assert.Equal("ancient", loaded.Name);
        Assert.Equal([1, 2], chain.Applied);
    }

    [Fact]
    public void the_stored_hash_is_never_compared_at_load()
    {
        // Python reads the hash and ignores it (src/versionable/_api.py:137-160): the hash is a
        // compile-time tripwire, and rejecting files by it would break every file written before a
        // schema change that the migration chain already handles.
        EngineVersioned loaded = Load<EngineVersioned>(
            EngineVersioned.Metadata,
            new() { ["name_v3"] = "fine" },
            version: 3,
            hash: "definitely-not-the-declared-hash");

        Assert.Equal("fine", loaded.Name);
    }

    // ------------------------------------------------------------------
    // Fields
    // ------------------------------------------------------------------

    [Fact]
    public void an_absent_field_falls_back_to_its_declared_default()
    {
        EngineSettings loaded = Load<EngineSettings>(
            EngineSettings.Metadata, new() { ["required"] = "present" }, version: 1);

        Assert.Equal("anon", loaded.Name);
        Assert.Equal(8080, loaded.Port);
    }

    [Fact]
    public void an_absent_field_with_no_default_is_a_clear_error()
    {
        BackendException error = Assert.Throws<BackendException>(
            () => Load<EngineSettings>(EngineSettings.Metadata, new() { ["name"] = "set" }, version: 1));

        Assert.Contains("'required' is missing", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void unknown_fields_are_dropped_by_default()
    {
        EngineSettings loaded = Load<EngineSettings>(
            EngineSettings.Metadata,
            new() { ["required"] = "present", ["removedLastYear"] = 1L },
            version: 1);

        Assert.Equal("present", loaded.Required);
    }

    [Fact]
    public void unknown_fields_are_reported_when_the_type_asks_for_it()
    {
        UnknownFieldException error = Assert.Throws<UnknownFieldException>(
            () => Load<EngineSettings>(
                EngineSettings.RejectingUnknown,
                new() { ["required"] = "present", ["removedLastYear"] = 1L, ["alsoGone"] = 2L },
                version: 1));

        Assert.Contains("[alsoGone, removedLastYear]", error.Message, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------
    // Literals
    // ------------------------------------------------------------------

    [Fact]
    public void a_declared_literal_option_loads()
    {
        EngineLiteralHolder loaded = Load<EngineLiteralHolder>(
            EngineLiteralHolder.Metadata, new() { ["mode"] = "slow", ["level"] = 2L }, version: 1);

        Assert.Equal("slow", loaded.Mode);
        Assert.Equal(2, loaded.Level);
    }

    [Fact]
    public void a_value_outside_the_declared_options_is_refused()
    {
        ConverterException error = Assert.Throws<ConverterException>(
            () => Load<EngineLiteralHolder>(
                EngineLiteralHolder.Metadata, new() { ["mode"] = "glacial", ["level"] = 2L }, version: 1));

        Assert.Contains("not a valid Literal option", error.Message, StringComparison.Ordinal);
        Assert.Contains("'fast', 'slow'", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_field_with_a_fallback_absorbs_an_unknown_value_and_warns()
    {
        List<string> warnings = [];
        void Capture(string message) => warnings.Add(message);
        VersionableLog.Warning += Capture;
        try
        {
            EngineLiteralHolder loaded = Load<EngineLiteralHolder>(
                EngineLiteralHolder.Metadata, new() { ["mode"] = "fast", ["level"] = 9L }, version: 1);

            Assert.Equal(1, loaded.Level);
        }
        finally
        {
            VersionableLog.Warning -= Capture;
        }

        Assert.Contains(warnings, message => message.Contains("Using the fallback", StringComparison.Ordinal));
    }

    [Fact]
    public void literal_validation_is_off_for_a_type_that_declares_it_off()
    {
        EngineLiteralHolder loaded = Load<EngineLiteralHolder>(
            EngineLiteralHolder.Unvalidated, new() { ["mode"] = "glacial", ["level"] = 9L }, version: 1);

        Assert.Equal("glacial", loaded.Mode);
        Assert.Equal(9, loaded.Level);
    }

    // ------------------------------------------------------------------
    // Save
    // ------------------------------------------------------------------

    [Fact]
    public void save_hands_the_backend_raw_values_and_the_envelope()
    {
        RecordingBackend backend = new([], new EnvelopeMetadata(null, null, null));

        VersionableFile.Save(new EngineLeaf("tip", 1.5), "memory.stub", backend);

        Assert.Equal("EngineLeaf", backend.SavedEnvelope?.ObjectName);
        Assert.Equal(1, backend.SavedEnvelope?.Version);
        Assert.Equal("aaaaaa", backend.SavedEnvelope?.Hash);
        Assert.Equal("tip", backend.SavedFields?["name"]);
        Assert.Equal(1.5, backend.SavedFields?["weight"]);
    }

    [Fact]
    public void skip_defaults_omits_fields_still_at_their_declared_value()
    {
        RecordingBackend backend = new([], new EnvelopeMetadata(null, null, null));
        EngineSettings settings = new("anon", 9000, "needed");

        VersionableFile.Save(settings, "memory.stub", backend);
        Assert.Equal(["name", "port", "required"], backend.SavedFields!.Keys.Order(StringComparer.Ordinal));

        SaveWith(EngineSettings.SkippingDefaults, settings, backend);
        Assert.Equal(["port", "required"], backend.SavedFields!.Keys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void saving_a_subclass_through_a_base_typed_call_writes_the_subclass_envelope()
    {
        RecordingBackend backend = new([], new EnvelopeMetadata(null, null, null));
        GoldenShape shape = new GoldenCircle { Label = "c1", Radius = 2.5 };

        VersionableFile.Save(shape, "memory.stub", backend);

        Assert.Equal("GoldenCircle", backend.SavedEnvelope?.ObjectName);
    }

    // ------------------------------------------------------------------
    // Backend option translation
    // ------------------------------------------------------------------

    [Fact]
    public void no_caller_preference_asks_the_backend_for_nothing_eagerly()
    {
        // BackendLoadOptions' own default means "materialize everything"; Python's load() with no
        // preload means "materialize no arrays". Passing the backend default straight through would
        // read every array of every HDF5 file nobody asked for.
        RecordingBackend backend = Loadable(new() { ["required"] = "present" }, version: 1);

        VersionableFile.Load<EngineSettings>("memory.stub", backend);

        Assert.NotNull(backend.LastLoadOptions?.Preload);
        Assert.Empty(backend.LastLoadOptions!.Preload!);
        Assert.Same(EngineSettings.Metadata, backend.LastLoadOptions.TargetMetadata);
    }

    [Fact]
    public void preload_all_asks_the_backend_for_everything()
    {
        RecordingBackend backend = Loadable(new() { ["required"] = "present" }, version: 1);

        VersionableFile.Load<EngineSettings>("memory.stub", backend, new VersionableLoadOptions { PreloadAll = true });

        Assert.Null(backend.LastLoadOptions?.Preload);
    }

    [Fact]
    public void an_explicit_preload_set_reaches_the_backend_unchanged()
    {
        RecordingBackend backend = Loadable(new() { ["required"] = "present" }, version: 1);
        FrozenSet<string> wanted = new[] { "required" }.ToFrozenSet(StringComparer.Ordinal);

        VersionableFile.Load<EngineSettings>(
            "memory.stub", backend, new VersionableLoadOptions { Preload = wanted, MetadataOnly = true });

        Assert.Equal(["required"], backend.LastLoadOptions?.Preload!);
        Assert.True(backend.LastLoadOptions?.MetadataOnly);
    }

    [Fact]
    public void a_lazy_field_reaches_the_instance_without_being_converted()
    {
        object sentinel = new();
        RecordingBackend backend = new(
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["either"] = sentinel },
            new EnvelopeMetadata("EngineUnionHolder", 1, "eeeeee"),
            new[] { "either" }.ToFrozenSet(StringComparer.Ordinal));

        EngineUnionHolder loaded = VersionableFile.Load<EngineUnionHolder>("memory.stub", backend);

        Assert.Same(sentinel, loaded.Either);
    }

    private static T Load<T>(
        VersionableMetadata metadata,
        Dictionary<string, object?> fields,
        int? version,
        string? hash = null,
        VersionableLoadOptions? options = null)
    {
        RecordingBackend backend = new(fields, new EnvelopeMetadata(metadata.Name, version, hash));
        return (T)LoadWith(metadata, backend, options);
    }

    private static object LoadWith(
        VersionableMetadata metadata,
        IVersionableBackend backend,
        VersionableLoadOptions? options)
    {
        // The public Load<T>() reaches metadata through T.VersionableMetadata, which is fixed per
        // type; these tests vary the metadata itself (chains, policies), so they register the
        // variant under the same Serialization Name and go through the dynamic entry point.
        VersionableRegistry.Reset();
        VersionableRegistry.Register(metadata);
        try
        {
            return VersionableFile.LoadDynamic("memory.stub", backend, options);
        }
        finally
        {
            VersionableRegistry.Reset();
            GoldenSchemas.EnsureRegistered();
        }
    }

    private static void SaveWith(VersionableMetadata metadata, object value, IVersionableBackend backend)
    {
        VersionableRegistry.Reset();
        VersionableRegistry.Register(metadata);
        try
        {
            VersionableFile.Save(value, "memory.stub", backend);
        }
        finally
        {
            VersionableRegistry.Reset();
            GoldenSchemas.EnsureRegistered();
        }
    }

    private static RecordingBackend Loadable(Dictionary<string, object?> fields, int version) =>
        new(fields, new EnvelopeMetadata("EngineSettings", version, "ffffff"));

    /// <summary>A backend that reads and writes nothing, and remembers what it was asked.</summary>
    private sealed class RecordingBackend(
        Dictionary<string, object?> fields,
        EnvelopeMetadata envelope,
        IReadOnlySet<string>? lazyFields = null) : IVersionableBackend
    {
        public IReadOnlySet<Type> NativeTypes => FrozenSet<Type>.Empty;

        public BackendLoadOptions? LastLoadOptions { get; private set; }

        public IReadOnlyDictionary<string, object?>? SavedFields { get; private set; }

        public EnvelopeMetadata? SavedEnvelope { get; private set; }

        public void Save(
            IReadOnlyDictionary<string, object?> savedFields,
            EnvelopeMetadata savedEnvelope,
            string path,
            VersionableMetadata metadata,
            BackendSaveOptions options)
        {
            SavedFields = savedFields;
            SavedEnvelope = savedEnvelope;
        }

        public BackendLoadResult Load(string path, BackendLoadOptions options)
        {
            LastLoadOptions = options;
            return new BackendLoadResult(fields, envelope, lazyFields);
        }
    }

}
