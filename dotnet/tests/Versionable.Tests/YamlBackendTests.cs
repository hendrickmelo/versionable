using System.Globalization;
using System.Text;
using Versionable.Backends;
using Versionable.Backends.Yaml;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The YAML backend against real files: what it writes, what it reads, and how it fails.
/// </summary>
[Collection(RegistryCollection.Name)]
public class YamlBackendTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"versionable-yaml-{Path.GetRandomFileName()}");

    public YamlBackendTests()
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

    [Fact]
    public void a_saved_object_loads_back_equal()
    {
        EngineNode node = new(
            "root",
            new EngineLeaf("centre", 2.5),
            [new EngineLeaf("left", 1.0), new EngineLeaf("right", 2.0)],
            new Dictionary<string, EngineLeaf>(StringComparer.Ordinal) { ["primary"] = new("primary", 3.5) },
            new HashSet<string>(["alpha", "beta"], StringComparer.Ordinal),
            new EngineLeaf("maybe", 0.25));
        string path = Path.Combine(_directory, "node.yaml");

        VersionableFile.Save(node, path);
        EngineNode loaded = VersionableFile.Load<EngineNode>(path);

        Assert.Equal(node.Label, loaded.Label);
        Assert.Equal(node.Leaf.Name, loaded.Leaf.Name);
        Assert.Equal(node.Leaf.Weight, loaded.Leaf.Weight);
        Assert.Equal(node.Leaves.Select(leaf => leaf.Name), loaded.Leaves.Select(leaf => leaf.Name));
        Assert.Equal(node.ByName["primary"].Weight, loaded.ByName["primary"].Weight);
        Assert.Equal(node.Tags.Order(StringComparer.Ordinal), loaded.Tags.Order(StringComparer.Ordinal));
        Assert.Equal(node.OptionalLeaf!.Name, loaded.OptionalLeaf!.Name);
    }

    /// <summary>Every golden fixture C# can write survives a save and load through this backend.</summary>
    /// <remarks>
    /// The corpus tests read Python's bytes; this writes C#'s and reads them back, which is the
    /// half that would catch a value the writer can emit but the reader cannot resolve — a string
    /// that looks like a number being the obvious one.
    /// </remarks>
    [Theory]
    [InlineData("scalars", "scalars.yaml")]
    [InlineData("containers", "containers.yaml")]
    [InlineData("optionals", "optionals.yaml")]
    [InlineData("enums", "enums.yaml")]
    [InlineData("literals", "literals.yaml")]
    [InlineData("nested", "nested.yaml")]
    [InlineData("polymorphic", "polymorphic.yaml")]
    [InlineData("temporal", "temporal.yaml")]
    [InlineData("stdlib", "stdlib.yaml")]
    [InlineData("arrays", "arrays.yaml")]
    [InlineData("migration-chain", "migration-chain.yaml")]
    public void a_golden_fixture_rewritten_by_csharp_reads_back_the_same(string fixture, string file)
    {
        string source = Path.Combine(GoldenRoot(), fixture, file);
        object original = VersionableFile.LoadDynamic(source);

        string rewritten = Path.Combine(_directory, file);
        VersionableFile.Save(original, rewritten);
        object reloaded = VersionableFile.LoadDynamic(rewritten);

        Assert.Equal(original.GetType(), reloaded.GetType());

        // Compared through a third format so the assertion does not depend on the fixture types
        // implementing equality, and so a value lost in YAML shows up as a JSON difference.
        Assert.Equal(AsJson(original), AsJson(reloaded));
    }

    /// <summary>
    /// Rewriting a golden fixture reproduces Python's bytes exactly.
    /// </summary>
    /// <remarks>
    /// Byte-identity is not the contract — the contract is that both languages read both files —
    /// but where it happens to hold it is the tightest available statement of writer parity, and
    /// it catches a quoting or float-formatting regression on the line it happens rather than
    /// three assertions later. Three fixtures are excluded and each for a reason that is not the
    /// emitter's: <c>arrays</c> because <c>json.dumps</c> puts a space after each comma and PyYAML
    /// folds the resulting scalar at 80 columns (see <see cref="EmbeddedNdarrayJson"/>), and
    /// <c>stdlib</c> and <c>containers</c> because their C# schema hashes legitimately differ
    /// (GRAMMAR §9 and §5), so the envelope line cannot match however the values are written.
    /// <para>
    /// If this fails after a deliberate change to the emitter, check that Python still reads the
    /// new output — <see cref="a_golden_fixture_rewritten_by_csharp_reads_back_the_same"/> and the
    /// corpus suite are the tests that must not be relaxed — and then update the fixture list here.
    /// </para>
    /// </remarks>
    /// <param name="fixture">Corpus directory name.</param>
    /// <param name="file">File within it.</param>
    [Theory]
    [InlineData("scalars", "scalars.yaml")]
    [InlineData("optionals", "optionals.yaml")]
    [InlineData("enums", "enums.yaml")]
    [InlineData("literals", "literals.yaml")]
    [InlineData("nested", "nested.yaml")]
    [InlineData("polymorphic", "polymorphic.yaml")]
    [InlineData("temporal", "temporal.yaml")]
    [InlineData("migration-chain", "migration-chain.yaml")]
    public void rewriting_a_golden_fixture_reproduces_pythons_bytes(string fixture, string file)
    {
        string source = Path.Combine(GoldenRoot(), fixture, file);
        string rewritten = Path.Combine(_directory, file);

        VersionableFile.Save(VersionableFile.LoadDynamic(source), rewritten);

        Assert.Equal(
            File.ReadAllText(source).ReplaceLineEndings("\n"),
            File.ReadAllText(rewritten).ReplaceLineEndings("\n"));
    }

    // ------------------------------------------------------------------
    // Layout
    // ------------------------------------------------------------------

    [Fact]
    public void the_envelope_is_written_last_and_holds_the_declared_schema()
    {
        string path = Path.Combine(_directory, "leaf.yaml");

        VersionableFile.Save(new EngineLeaf("tip", 1.5), path);

        // Python's YAML backend appends the envelope after the fields, where its JSON backend puts
        // it first. Both golden corpora say so and both are read here, so the writer has to pick
        // the right one per format.
        Assert.Equal(
            """
            name: tip
            weight: 1.5
            __versionable__:
              object: EngineLeaf
              version: 1
              hash: aaaaaa

            """,
            File.ReadAllText(path).ReplaceLineEndings("\n"));
    }

    [Fact]
    public void the_file_is_utf8_without_a_byte_order_mark()
    {
        string path = Path.Combine(_directory, "leaf.yaml");

        VersionableFile.Save(new EngineLeaf("é", 1.5), path);

        byte[] bytes = File.ReadAllBytes(path);
        Assert.NotEqual<byte[]>([0xEF, 0xBB, 0xBF], bytes.Take(3).ToArray());
        Assert.Equal("é", VersionableFile.Load<EngineLeaf>(path).Name);
    }

    [Fact]
    public void fields_are_written_in_declaration_order()
    {
        string path = Path.Combine(_directory, "settings.yaml");

        VersionableFile.Save(new EngineSettings("named", 1234, "needed"), path);

        Assert.Equal(
            ["name", "port", "required", VersionableEnvelope.WrappedKey],
            File.ReadLines(path)
                .Where(line => line.Length > 0 && !char.IsWhiteSpace(line[0]))
                .Select(line => line[..line.IndexOf(':', StringComparison.Ordinal)]));
    }

    [Theory]
    [InlineData(".yaml")]
    [InlineData(".yml")]
    public void both_extensions_reach_this_backend(string extension)
    {
        string path = Path.Combine(_directory, $"leaf{extension}");

        VersionableFile.Save(new EngineLeaf("tip", 1.5), path);

        Assert.Equal("tip", VersionableFile.Load<EngineLeaf>(path).Name);
    }

    [Fact]
    public void the_dynamic_entry_point_resolves_the_type_from_the_envelope()
    {
        string path = Path.Combine(_directory, "leaf.yaml");
        VersionableFile.Save(new EngineLeaf("tip", 1.5), path);

        Assert.Equal("tip", Assert.IsType<EngineLeaf>(VersionableFile.LoadDynamic(path)).Name);
    }

    // ------------------------------------------------------------------
    // PyYAML's schema, on the way out
    // ------------------------------------------------------------------

    /// <summary>
    /// A string that would read back as some other type is quoted.
    /// </summary>
    /// <remarks>
    /// The single most important thing this backend does that a generic YAML writer does not.
    /// Every one of these values is a legitimate string in a schema and every one of them is a
    /// different type unquoted — under YAML 1.1, which is the schema PyYAML applies to files this
    /// writes. Getting it wrong is silent: the file still parses, the field just changes type.
    /// </remarks>
    [Theory]
    [InlineData("660511", "'660511'")]
    [InlineData("yes", "'yes'")]
    [InlineData("Off", "'Off'")]
    [InlineData("true", "'true'")]
    [InlineData("null", "'null'")]
    [InlineData("~", "'~'")]
    [InlineData("", "''")]
    [InlineData("12.50", "'12.50'")]
    [InlineData(".inf", "'.inf'")]
    [InlineData("0x1f", "'0x1f'")]
    [InlineData("012", "'012'")]
    [InlineData("1_000", "'1_000'")]
    [InlineData("2026-08-05", "'2026-08-05'")]
    [InlineData("12:30:15", "'12:30:15'")]
    [InlineData("<<", "'<<'")]
    // And the ones that must NOT be quoted, or the file stops looking hand-written.
    [InlineData("probe-A", "probe-A")]
    [InlineData("0o17", "0o17")]
    [InlineData("1e5", "1e5")]
    [InlineData("08", "08")]
    [InlineData("C:\\Devices\\probe.cfg", "C:\\Devices\\probe.cfg")]
    public void a_string_is_quoted_exactly_when_pyyaml_would_read_it_as_something_else(
        string value,
        string expected)
    {
        string path = Path.Combine(_directory, "leaf.yaml");

        VersionableFile.Save(new EngineLeaf(value, 1.5), path);

        Assert.Equal($"name: {expected}", File.ReadLines(path).First());
        Assert.Equal(value, VersionableFile.Load<EngineLeaf>(path).Name);
    }

    [Theory]
    [InlineData("line one\nline two")]
    [InlineData("trailing space ")]
    [InlineData("  leading space")]
    [InlineData("key: value")]
    [InlineData("- not a list")]
    [InlineData("#not a comment")]
    [InlineData("a very long string that runs past the eighty columns PyYAML would fold at, to make "
        + "sure nothing here silently reflows it into a different value")]
    public void a_string_whose_shape_collides_with_yaml_syntax_survives(string value)
    {
        // These are the emitter's own analysis to get right rather than the schema's, but a
        // backend that only ever wrote field-name-shaped strings would never find out.
        string path = Path.Combine(_directory, "awkward.yaml");

        VersionableFile.Save(new EngineLeaf(value, 1.5), path);

        Assert.Equal(value, VersionableFile.Load<EngineLeaf>(path).Name);
    }

    [Fact]
    public void a_whole_float_keeps_its_point_and_a_lower_case_exponent()
    {
        // `weight: 2` would come back an integer, and an exponent with no point (`1e+30`) does not
        // even match PyYAML's float pattern — it would come back a string. The lower-case `e` is
        // PyYAML's spelling: `yaml.dump({"weight": 1e30})` writes `weight: 1.0e+30`, and .NET's
        // "R" format writes `1E+30`.
        string path = Path.Combine(_directory, "leaf.yaml");

        VersionableFile.Save(new EngineLeaf("tip", 2.0), path);
        Assert.Contains("weight: 2.0", File.ReadAllText(path), StringComparison.Ordinal);

        VersionableFile.Save(new EngineLeaf("tip", 1e30), path);
        Assert.Contains("weight: 1.0e+30", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Equal(1e30, VersionableFile.Load<EngineLeaf>(path).Weight);

        // A mantissa that already has a point keeps it; only the exponent's case changes.
        VersionableFile.Save(new EngineLeaf("tip", 1.5e-20), path);
        Assert.Contains("weight: 1.5e-20", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Equal(1.5e-20, VersionableFile.Load<EngineLeaf>(path).Weight);
    }

    [Theory]
    [InlineData(double.NaN, ".nan")]
    [InlineData(double.PositiveInfinity, ".inf")]
    [InlineData(double.NegativeInfinity, "-.inf")]
    public void non_finite_numbers_are_written_with_yaml_spellings(double value, string token)
    {
        // Unlike JSON, YAML has spellings for these, so there is no repair pass on either side.
        string path = Path.Combine(_directory, "nonfinite.yaml");

        VersionableFile.Save(new EngineLeaf("tip", value), path);

        Assert.Contains($"weight: {token}", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Equal(value, VersionableFile.Load<EngineLeaf>(path).Weight);
    }

    [Fact]
    public void integers_survive_the_round_trip_at_full_width()
    {
        string path = Path.Combine(_directory, "numbers.yaml");
        EngineNumbers numbers = new(long.MaxValue, int.MinValue, ulong.MaxValue, 9.8125);

        VersionableFile.Save(numbers, path);
        EngineNumbers loaded = VersionableFile.Load<EngineNumbers>(path);

        Assert.Equal(long.MaxValue, loaded.Big);
        Assert.Equal(int.MinValue, loaded.Small);
        Assert.Equal(ulong.MaxValue, loaded.Huge);
        Assert.Equal(9.8125, loaded.Ratio);
    }

    [Fact]
    public void an_integer_past_the_exact_range_of_a_double_keeps_every_bit()
    {
        const long beyondDoublePrecision = (1L << 53) + 1;
        string path = Path.Combine(_directory, "precise.yaml");

        VersionableFile.Save(new EngineNumbers(beyondDoublePrecision, 0, 0, 0.5), path);

        Assert.Contains("9007199254740993", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Equal(beyondDoublePrecision, VersionableFile.Load<EngineNumbers>(path).Big);
    }

    // ------------------------------------------------------------------
    // PyYAML's schema, on the way in
    // ------------------------------------------------------------------

    /// <summary>
    /// A hand-written file is read with PyYAML's tolerances, not YAML 1.2's.
    /// </summary>
    /// <remarks>
    /// Every spelling here is one a person might type into a config file, and Python already
    /// answers for each of them. <c>0o17</c> and <c>08</c> are the interesting pair: 1.2 reads the
    /// first as octal 15 and PyYAML reads it as the string, and neither language accepts <c>08</c>
    /// as a number at all.
    /// </remarks>
    [Theory]
    [InlineData("42", 42.0)]
    [InlineData("012", 10.0)]
    [InlineData("0x1f", 31.0)]
    [InlineData("0b101", 5.0)]
    [InlineData("1_000", 1000.0)]
    [InlineData("1:30", 90.0)]
    [InlineData("-1.5e+3", -1500.0)]
    [InlineData(".inf", double.PositiveInfinity)]
    [InlineData("-.inf", double.NegativeInfinity)]
    [InlineData("190:20:30.5", 685230.5)]
    public void a_number_written_by_hand_is_read_the_way_pyyaml_reads_it(string token, double expected)
    {
        string path = Path.Combine(_directory, "hand.yaml");
        WriteText(path, $"name: tip\nweight: {token}\n{Envelope("EngineLeaf", "aaaaaa")}");

        Assert.Equal(expected, VersionableFile.Load<EngineLeaf>(path).Weight);
    }

    [Theory]
    // `yes` is a bool to YAML 1.1, and a bool reaching a str field stringifies — to `True`, which
    // is what Python's `deserialize(True, str)` returns as well. Quote it in the file to get the
    // word back; the writer always does.
    [InlineData("yes", "True")]
    [InlineData("0o17", "0o17")]
    [InlineData("08", "08")]
    [InlineData("1e5", "1e5")]
    [InlineData("2026-08-05", "2026-08-05")]
    [InlineData("!!str 42", "42")]
    [InlineData("'42'", "42")]
    [InlineData("\"true\"", "true")]
    public void a_scalar_that_is_not_a_string_to_yaml_still_lands_in_a_string_field(
        string token,
        string expected)
    {
        // The reader hands back whatever the schema says and the engine converts; the point is
        // that nothing is lost on the way. `2026-08-05` is the one that matters: read as a date it
        // would reach a str field as a formatted date rather than the text the file holds.
        string path = Path.Combine(_directory, "hand.yaml");
        WriteText(path, $"name: {token}\nweight: 1.5\n{Envelope("EngineLeaf", "aaaaaa")}");

        Assert.Equal(expected, VersionableFile.Load<EngineLeaf>(path).Name);
    }

    [Fact]
    public void a_timestamp_reaches_a_temporal_field_as_the_text_the_file_holds()
    {
        // Python's writer quotes these, so the round trip never depends on it — but a person
        // editing the file will drop the quotes, and PyYAML then constructs a datetime. Keeping
        // the text is what makes both spellings load identically here.
        string path = Path.Combine(_directory, "hand.yaml");
        WriteText(
            path,
            "naive: 2026-08-05T14:30:15.123456\naware: '2026-08-05T14:30:15.123456-05:00'\n"
                + "day: 2026-08-05\nclock: '23:59:58.500000'\nelapsed: 90061.5\n"
                + Envelope("GoldenTemporal", "3856a2"));

        GoldenTemporal loaded = VersionableFile.Load<GoldenTemporal>(path);

        Assert.Equal(new DateTime(2026, 8, 5, 14, 30, 15, DateTimeKind.Unspecified).AddTicks(1234560), loaded.Naive);
        Assert.Equal(new DateOnly(2026, 8, 5), loaded.Day);
    }

    [Fact]
    public void an_alias_resolves_to_the_node_its_anchor_named()
    {
        // Nothing versionable writes emits an anchor — the walker builds a fresh mapping for every
        // nested object, so PyYAML finds no shared reference to alias. A hand-written or
        // hand-merged file is another matter, and Python reads those.
        string path = Path.Combine(_directory, "anchored.yaml");
        WriteText(
            path,
            """
            label: assembly
            inner: &origin
              __versionable__: {object: GoldenInner, version: 1, hash: e37514}
              x: 1.5
              y: -2.5
            points: [*origin]
            byName: {only: *origin}
            optionalInner: *origin
            __versionable__: {object: GoldenNested, version: 1, hash: 705e61}
            """);

        GoldenNested loaded = VersionableFile.Load<GoldenNested>(path);

        Assert.Equal(1.5, loaded.Inner.X);
        Assert.Equal(1.5, Assert.Single(loaded.Points).X);
        Assert.Equal(-2.5, loaded.ByName["only"].Y);
        Assert.Equal(1.5, loaded.OptionalInner!.X);
    }

    [Fact]
    public void a_merge_key_pulls_in_another_mapping_and_loses_to_what_is_written_explicitly()
    {
        string path = Path.Combine(_directory, "merged.yaml");
        WriteText(
            path,
            """
            defaults: &defaults
              name: anon
              port: 8080
              required: needed
            settings:
              <<: *defaults
              port: 9000
            """);

        // Read through the backend rather than a schema: `defaults` is not a field of anything,
        // and the point is what the mapping under `settings` came out as.
        BackendLoadResult result = new YamlBackend().Load(path, new BackendLoadOptions());
        Dictionary<string, object?> settings =
            Assert.IsType<Dictionary<string, object?>>(result.Fields["settings"]);

        Assert.Equal("anon", settings["name"]);
        Assert.Equal("needed", settings["required"]);
        Assert.Equal(9000L, settings["port"]);
    }

    [Fact]
    public void a_file_written_by_python_in_the_legacy_layout_still_loads()
    {
        string path = Path.Combine(_directory, "legacy.yaml");
        WriteText(
            path,
            """
            __OBJECT__: EngineLeaf
            __VERSION__: 1
            __HASH__: aaaaaa
            name: from-0.1.x
            weight: 2.0
            """);

        EngineLeaf leaf = VersionableFile.Load<EngineLeaf>(path);

        Assert.Equal("from-0.1.x", leaf.Name);
        Assert.Equal(2.0, leaf.Weight);
    }

    /// <summary>
    /// An array this backend writes lands in the <c>__ver_json__</c> wrapper, not as a YAML
    /// mapping.
    /// </summary>
    /// <remarks>
    /// The write side of <see cref="YamlGoldenCorpusTests.the_corpus_stores_an_array_field_as_a_single_embedded_json_wrapper"/>,
    /// which asserts the same shape on Python's bytes. A writer that emitted the four-key ndarray
    /// mapping directly would still round-trip through this backend and would still pass every
    /// value assertion here — and Python would refuse the file, because
    /// <c>_fromYamlSafe</c> only unwraps the wrapper.
    /// </remarks>
    [Fact]
    public void an_array_field_written_by_csharp_carries_the_embedded_json_wrapper()
    {
        string path = Path.Combine(_directory, "arrays.yaml");
        VersionableFile.Save(
            VersionableFile.LoadDynamic(Path.Combine(GoldenRoot(), "arrays", "arrays.yaml")),
            path);

        string text = File.ReadAllText(path).ReplaceLineEndings("\n");

        // Compact separators, unlike Python's `json.dumps` — see EmbeddedNdarrayJson, which is why
        // `arrays` is the one fixture excluded from the byte-identity theory above.
        Assert.Contains(
            "signal:\n  __ver_json__: '{\"__ver_ndarray__\":true,\"dtype\":\"float64\"",
            text,
            StringComparison.Ordinal);

        // The payload stays a string: an unwrapped mapping would put these on their own lines.
        Assert.DoesNotContain("\n    dtype:", text, StringComparison.Ordinal);
        Assert.DoesNotContain("\n    shape:", text, StringComparison.Ordinal);

        // And an array nested inside a container is wrapped too, not just a top-level field.
        Assert.Contains("channels:\n  ch0:\n    __ver_json__:", text, StringComparison.Ordinal);
    }

    [Fact]
    public void the_legacy_embedded_json_wrapper_is_still_unwrapped()
    {
        // 0.1.x wrote `__json__` where 0.2 writes `__ver_json__`; Python still reads both.
        string source = Path.Combine(GoldenRoot(), "arrays", "arrays.yaml");
        string path = Path.Combine(_directory, "legacy-arrays.yaml");
        WriteText(
            path,
            File.ReadAllText(source).Replace("__ver_json__:", "__json__:", StringComparison.Ordinal));

        Assert.Equal(4, VersionableFile.Load<GoldenArrays>(path).Signal.FlattenedLength);
    }

    // ------------------------------------------------------------------
    // commentDefaults
    // ------------------------------------------------------------------

    [Fact]
    public void comment_defaults_comments_out_the_fields_still_at_their_default()
    {
        string path = Path.Combine(_directory, "settings.yaml");

        VersionableFile.Save(
            new EngineSettings("anon", 9000, "needed"),
            path,
            options: new BackendSaveOptions { CommentDefaults = true });

        Assert.Equal(
            """
            # name: anon
            port: 9000
            required: needed
            __versionable__:
              object: EngineSettings
              version: 1
              hash: ffffff

            """,
            File.ReadAllText(path).ReplaceLineEndings("\n"));
    }

    [Fact]
    public void a_commented_out_field_comes_back_as_its_default()
    {
        // The whole point: the file is still loadable and the field is still what it was.
        string path = Path.Combine(_directory, "settings.yaml");

        VersionableFile.Save(
            new EngineSettings("anon", 8080, "needed"),
            path,
            options: new BackendSaveOptions { CommentDefaults = true });
        EngineSettings loaded = VersionableFile.Load<EngineSettings>(path);

        Assert.Equal("anon", loaded.Name);
        Assert.Equal(8080, loaded.Port);
        Assert.Equal("needed", loaded.Required);
    }

    [Fact]
    public void comment_defaults_never_comments_out_the_envelope()
    {
        string path = Path.Combine(_directory, "leaf.yaml");

        VersionableFile.Save(
            new EngineLeaf("", 0.0),
            path,
            options: new BackendSaveOptions { CommentDefaults = true });

        // Both fields are at their default, so without the exception the file would be entirely
        // comments and would load as a version-less object of no declared type.
        Assert.StartsWith("# name:", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Equal("EngineLeaf", VersionableFile.LoadDynamic(path).GetType().Name);
    }

    // ------------------------------------------------------------------
    // Failures
    // ------------------------------------------------------------------

    [Fact]
    public void malformed_yaml_is_reported_as_a_backend_failure()
    {
        string path = Path.Combine(_directory, "broken.yaml");
        WriteText(path, "name: tip\n  weight: [1, 2\n");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("Failed to read YAML", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_yaml_document_that_is_not_a_mapping_is_reported()
    {
        string path = Path.Combine(_directory, "sequence.yaml");
        WriteText(path, "- 1\n- 2\n");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("Expected a YAML mapping", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_empty_file_is_reported_rather_than_loaded_as_an_empty_object()
    {
        string path = Path.Combine(_directory, "empty.yaml");
        WriteText(path, string.Empty);

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("Expected a YAML mapping", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_second_document_in_the_stream_is_refused()
    {
        // `yaml.safe_load` raises on a multi-document stream too; taking the first would silently
        // drop whatever the second one holds.
        string path = Path.Combine(_directory, "two.yaml");
        WriteText(path, $"name: first\nweight: 1.0\n{Envelope("EngineLeaf", "aaaaaa")}---\nname: second\nweight: 2.0\n");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("more than one YAML document", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_missing_file_is_reported_as_a_backend_failure()
    {
        BackendException error = Assert.Throws<BackendException>(
            () => VersionableFile.Load<EngineLeaf>(Path.Combine(_directory, "absent.yaml")));

        Assert.Contains("Failed to read YAML", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_undefined_anchor_is_reported()
    {
        string path = Path.Combine(_directory, "dangling.yaml");
        WriteText(path, "name: *nowhere\nweight: 1.0\n");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("undefined anchor", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_corrupt_envelope_table_is_reported_rather_than_ignored()
    {
        string path = Path.Combine(_directory, "corrupt.yaml");
        WriteText(path, "name: x\nweight: 1.0\n__versionable__: GoldenInner\n");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("must hold the envelope table", error.Message, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------
    // Contract
    // ------------------------------------------------------------------

    [Fact]
    public void yaml_stores_nothing_natively_so_every_value_goes_through_the_walker()
    {
        Assert.Empty(new YamlBackend().NativeTypes);
    }

    [Fact]
    public void yaml_never_reports_a_lazy_field()
    {
        string path = Path.Combine(_directory, "leaf.yaml");
        VersionableFile.Save(new EngineLeaf("tip", 1.5), path);

        BackendLoadResult result = new YamlBackend().Load(path, new BackendLoadOptions { MetadataOnly = true });

        Assert.Null(result.LazyFields);
        Assert.Equal("tip", result.Fields["name"]);
    }

    [Fact]
    public void an_explicit_backend_overrides_the_extension()
    {
        string path = Path.Combine(_directory, "leaf.unknown");

        VersionableFile.Save(new EngineLeaf("tip", 1.5), path, new YamlBackend());

        Assert.Equal("tip", VersionableFile.Load<EngineLeaf>(path, new YamlBackend()).Name);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static string Envelope(string name, string hash) =>
        $"__versionable__:\n  object: {name}\n  version: 1\n  hash: {hash}\n";

    private static void WriteText(string path, string content) =>
        File.WriteAllText(path, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

    private static string AsJson(object value)
    {
        string path = Path.Combine(Path.GetTempPath(), $"versionable-yaml-{Path.GetRandomFileName()}.json");
        try
        {
            VersionableFile.Save(value, path);
            return File.ReadAllText(path);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string GoldenRoot()
    {
        for (DirectoryInfo? directory = new(AppContext.BaseDirectory);
            directory is not null;
            directory = directory.Parent)
        {
            string candidate = Path.Combine(directory.FullName, "conformance", "golden");
            if (File.Exists(Path.Combine(candidate, "index.json")))
            {
                return candidate;
            }
        }

        throw new InvalidOperationException(
            string.Create(
                CultureInfo.InvariantCulture,
                $"conformance/golden not found above '{AppContext.BaseDirectory}'."));
    }
}
