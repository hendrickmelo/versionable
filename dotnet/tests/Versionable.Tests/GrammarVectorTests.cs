using System.Text.Json;
using Versionable.Analyzers.Grammar;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Drives every vector in <c>conformance/hash-vectors.json</c> through the Roslyn renderer.
/// </summary>
/// <remarks>
/// A vector passes only when a real C# declaration renders to the vector's canonical strings,
/// rebuilds its payload, and reproduces its hash — checking <c>payload</c> against
/// <c>hash</c> alone would test SHA-256, not the grammar (GRAMMAR.md §13). The vector file is
/// read from the repository rather than copied, so a vector change lands here on the next
/// run instead of drifting.
/// </remarks>
public class GrammarVectorTests
{
    /// <summary>
    /// Vector name to the C# members whose canonical rendering the vector describes.
    /// Public fields, because what is under test is the type rendering, not the member shape.
    /// </summary>
    private static readonly Dictionary<string, string> _fixtures = new(StringComparer.Ordinal)
    {
        ["no-fields"] = "",
        ["single-field"] = "public int value;",
        ["simple-scalars"] = "public int count; public string label;",
        ["field-name-sorting"] = "public int zeta; public string alpha; public bool Mid;",
        ["field-name-sorting-case"] = "public int b; public int B; public int a; public int A;",
        ["worked-example"] = "public string label; public int count; public HashSet<string> tags;",
        ["bool-is-not-int"] = "public bool flag;",
        ["scalar-width-erased"] = "public long big; public byte small; public double precise; public float rough;",
        ["complex-and-bytes"] = "public byte[] blob; public Complex z;",
        ["list-of-int"] = "public List<int> items;",
        ["dict-str-float"] = "public Dictionary<string, double> table;",
        ["set-of-str"] = "public HashSet<string> tags;",
        ["frozenset-of-int"] = "public FrozenSet<int> ids;",
        ["tuple-mixed"] = "public (int, string, double) row;",
        ["tuple-mixed-reordered"] = "public (string, int, double) row;",
        ["tuple-single"] = "public ValueTuple<int> pair;",
        ["nested-containers"] =
            "public List<List<double>> matrix; public Dictionary<string, List<int>> index;",
        ["deep-nesting"] = "public List<Dictionary<string, (int, double)>> grid;",
        ["container-of-serialization-name"] =
            "public List<Node> nodes; public Dictionary<Status, int> byStatus; public HashSet<Status> flags;",
        ["optional-int"] = "public int? value;",
        ["union-none-not-first"] = "public decimal? value;",
        ["union-in-container"] = "public Dictionary<string, double?> lookup;",
        ["enum-bare-name"] = "public Status status;",
        ["enum-explicit-name"] = "public LocalStatus status;",
        ["versionable-name"] = "public DeviceConfig config;",
        ["versionable-forward-ref"] = "public List<Node> children; public Node? parent;",
        ["datetime-family"] =
            "public DateTime created; public DateOnly day; public TimeOnly clock; public TimeSpan elapsed;",
        ["stdlib-converters"] =
            "public FilePath path; public decimal amount; public Guid id; public Regex pattern;",
        ["ndarray-float64"] = "public Tensor<double> data;",
        ["ndarray-float32"] = "public Tensor<float> data;",
        ["ndarray-int32"] = "public Tensor<int> counts;",
        ["ndarray-int64"] = "public Tensor<long> counts;",
        ["ndarray-uint8"] = "public Tensor<byte> image;",
        ["ndarray-bool"] = "public Tensor<bool> mask;",
        ["ndarray-complex128"] = "public Tensor<Complex> spectrum;",
        ["ndarray-float16"] = "public Tensor<Half> weights;",

        // C# has no shape in the type, so the 3-D declaration is spelled the same as the 1-D
        // one. Sharing the fixture is the point: shape erasure is structural here.
        ["ndarray-shape-erased"] = "public Tensor<double> data;",
        ["ndarray-in-container"] =
            "public List<Tensor<byte>> frames; public Dictionary<string, Tensor<double>> byName;",
        ["ndarray-optional"] = "public Tensor<double>? data;",
        ["literal-strings"] = """[LiteralValues("fast", "slow")] public string mode;""",
        ["literal-strings-reordered"] = """[LiteralValues("slow", "fast")] public string mode;""",
        ["literal-ints"] = "[LiteralValues(1, 2, 3)] public int level;",
        ["literal-str-one"] = """[LiteralValues("1")] public string value;""",
        ["literal-int-one"] = "[LiteralValues(1)] public int value;",
        ["literal-mixed"] = """[LiteralValues("auto", 0, "off", 1)] public object tag;""",
        ["literal-negative-int"] = "[LiteralValues(-1, 0, 1)] public int offset;",
        ["literal-single-member"] = """[LiteralValues("only")] public string kind;""",
        ["literal-optional"] = """[LiteralValues("fast", "slow")] public string? mode;""",
        ["literal-bool"] = "[LiteralValues(true, false)] public bool flag;",
        ["literal-bool-vs-int"] = "[LiteralValues(1, 0)] public int flag;",
        ["literal-escapes"] = """
            [LiteralValues("it's")] public string quote;
            [LiteralValues("a\\b")] public string backslash;
            """,

        // The C# counterpart of Annotated[...]: a metadata attribute that must not reach the hash.
        ["annotated-unwrapped"] = """[Description("dB")] public double gain; public string note;""",
        ["annotated-nested"] = "public List<double> samples;",
        ["realistic-measurement"] = """
            public string name;
            public DateTime acquiredAt;
            public Tensor<double> samples;
            public double sampleRate_Hz;
            [LiteralValues("fast", "slow")] public string mode;
            public Status status;
            public string? @operator;
            public HashSet<string> tags;
            public Calibration? calibration;
            """,
        ["realistic-device-config"] = """
            public Guid deviceId;
            public FilePath firmwarePath;
            public TimeSpan timeout;
            public Dictionary<string, (double, double)> limits;
            public List<ChannelConfig> channels;
            public Regex serialPattern;
            public decimal price;
            public byte[] raw;
            """,
        ["sort-by-name-not-pair"] = "public int a1; public string a;",
        ["parameterized-non-container"] = "public Regex rx; public MyBox<int> box;",
        ["non-ascii-utf8"] = """
            public int größe;
            [LiteralValues("café", "日本語")] public string label;
            """,
    };

    /// <summary>Vector names the C# suite must render.</summary>
    /// <returns>One row per renderable vector.</returns>
    public static TheoryData<string> RenderableVectors()
    {
        TheoryData<string> data = new();
        foreach (Vector vector in _vectors.Value.Where(vector => _fixtures.ContainsKey(vector.Name)))
        {
            data.Add(vector.Name);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(RenderableVectors))]
    public void vector_renders_payload_and_hash(string name)
    {
        Vector vector = _vectors.Value.Single(candidate => candidate.Name == name);
        SchemaModel schema = GrammarTestHarness.Schema(_fixtures[name]);

        Assert.Empty(schema.Problems.Where(problem => problem.Id != "VSN0001"));

        // Field-by-field first: a payload mismatch alone would not say which type drifted.
        Dictionary<string, string> rendered = schema.Fields.ToDictionary(
            field => field.WireName,
            field => field.CanonicalType,
            StringComparer.Ordinal);
        Assert.Equal(
            vector.Fields.OrderBy(pair => pair.Key, StringComparer.Ordinal),
            rendered.OrderBy(pair => pair.Key, StringComparer.Ordinal));

        Assert.Equal(vector.Payload, schema.Payload);
        Assert.Equal(vector.Hash, schema.ComputedHash);
    }

    [Fact]
    public void every_vector_is_either_rendered_or_flagged_undeclarable()
    {
        // The exhaustiveness guard. A new vector that C# can declare must arrive with a
        // fixture; one it cannot must arrive carrying "csharpDeclarable": false, which is data
        // in the vector file rather than a list living in this test.
        List<string> unaccounted = _vectors.Value
            .Where(vector => !vector.PythonOnly
                && vector.CsharpDeclarable
                && !_fixtures.ContainsKey(vector.Name))
            .Select(vector => vector.Name)
            .ToList();

        Assert.Empty(unaccounted);
    }

    [Fact]
    public void an_undeclarable_vector_is_undeclarable_for_one_documented_reason()
    {
        // C# spells optionality as T? — a two-member union that null short-circuits — and has
        // no type-level spelling for three or more members. Nothing else in grammar version 1
        // is unreachable, so the flag is expected to name exactly the n-ary unions.
        List<string> flagged = _vectors.Value
            .Where(vector => !vector.CsharpDeclarable)
            .Select(vector => vector.Name)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToList();

        Assert.Equal(
            new[] { "union-nested", "union-none-middle", "union-of-serialization-names", "union-sorting" },
            flagged);
        // Every one of them is a union of three or more members — the only form C# cannot
        // spell. Counting the separators inside the brackets is what says so.
        Assert.All(
            _vectors.Value.Where(vector => !vector.CsharpDeclarable),
            vector =>
            {
                Assert.Contains(":Union[", vector.Payload, StringComparison.Ordinal);
                Assert.True(
                    vector.Payload.Split(", ").Length >= 3,
                    $"{vector.Name} is flagged undeclarable but is not an n-ary union: {vector.Payload}");
            });

        // pythonOnly and csharpDeclarable describe different things and never overlap.
        Assert.DoesNotContain(_vectors.Value, vector => vector.PythonOnly && !vector.CsharpDeclarable);
    }

    [Fact]
    public void must_match_and_must_differ_hold_for_the_hashes_this_renderer_produces()
    {
        // The relationships are only worth anything if they hold for what C# *renders*, not
        // just for what the file already says. Where both sides have a fixture, both hashes are
        // recomputed from real declarations: ndarray-float64 and ndarray-shape-erased must
        // collide (shape erasure) while ndarray-float32 must not (dtype significance, ADR-0002).
        Dictionary<string, Vector> byName = _vectors.Value.ToDictionary(
            vector => vector.Name,
            StringComparer.Ordinal);
        int rendered = 0;

        foreach (Vector vector in _vectors.Value)
        {
            foreach (string other in vector.MustMatch)
            {
                Assert.Equal(byName[other].Hash, vector.Hash);
                if (TryRender(vector, out string left) && TryRender(byName[other], out string right))
                {
                    Assert.Equal(right, left);
                    rendered++;
                }
            }

            foreach (string other in vector.MustDiffer)
            {
                Assert.NotEqual(byName[other].Hash, vector.Hash);
                if (TryRender(vector, out string left) && TryRender(byName[other], out string right))
                {
                    Assert.NotEqual(right, left);
                    rendered++;
                }
            }
        }

        // Guards the guard: if the fixture table ever stops covering these pairs, the loop
        // above would silently degrade to the file-consistency check it replaced.
        Assert.True(rendered >= 10, $"only {rendered} relationships were checked against rendered hashes");
    }

    private static bool TryRender(Vector vector, out string hash)
    {
        hash = string.Empty;
        if (!_fixtures.TryGetValue(vector.Name, out string? members))
        {
            return false;
        }

        hash = GrammarTestHarness.Schema(members).ComputedHash;
        return true;
    }

    [Fact]
    public void the_vector_file_declares_grammar_version_one()
    {
        // A bump means hashes may have moved, so the C# renderer must be revisited in the
        // same generation (ADR-0004). Failing here is the tripwire for that.
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(VectorPath));

        Assert.Equal(1, document.RootElement.GetProperty("grammarVersion").GetInt32());
    }

    private static string VectorPath =>
        Path.Combine(GrammarTestHarness.RepositoryRoot(), "conformance", "hash-vectors.json");

    private static readonly Lazy<IReadOnlyList<Vector>> _vectors = new(() =>
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(VectorPath));

        return document.RootElement.GetProperty("vectors").EnumerateArray().Select(element => new Vector(
            element.GetProperty("name").GetString()!,
            element.GetProperty("payload").GetString()!,
            element.GetProperty("hash").GetString()!,
            element.TryGetProperty("pythonOnly", out JsonElement pythonOnly) && pythonOnly.GetBoolean(),
            !element.TryGetProperty("csharpDeclarable", out JsonElement declarable) || declarable.GetBoolean(),
            element.GetProperty("fields").EnumerateObject().ToDictionary(
                field => field.Name,
                field => field.Value.GetString()!,
                StringComparer.Ordinal),
            Names(element, "mustMatch"),
            Names(element, "mustDiffer"))).ToList();

        static IReadOnlyList<string> Names(JsonElement element, string key) =>
            element.TryGetProperty(key, out JsonElement names)
                ? names.EnumerateArray().Select(name => name.GetString()!).ToList()
                : Array.Empty<string>();
    });

    private sealed record Vector(
        string Name,
        string Payload,
        string Hash,
        bool PythonOnly,
        bool CsharpDeclarable,
        IReadOnlyDictionary<string, string> Fields,
        IReadOnlyList<string> MustMatch,
        IReadOnlyList<string> MustDiffer);
}
