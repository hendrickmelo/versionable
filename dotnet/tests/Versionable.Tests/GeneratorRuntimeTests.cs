using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Emit;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Compiles a fixture through the generator, loads the result, and exercises the generated
/// metadata: the factory, the accessors, and the module-initializer registration.
/// </summary>
/// <remarks>
/// Each fixture is loaded into its own <see cref="AssemblyLoadContext"/>, which resolves its
/// own copy of <c>Versionable.dll</c>. That gives the fixture a private
/// <c>VersionableRegistry</c>: registration is a process-wide static, and asserting on the
/// test host's copy would race every other test class that resets it. Delegates still cross
/// the boundary, because <c>Func&lt;,&gt;</c> comes from the shared runtime.
/// </remarks>
public class GeneratorRuntimeTests
{
    [Fact]
    public void generated_metadata_describes_the_type_and_rebuilds_it()
    {
        object metadata = Metadata("RuntimeConfig", $$"""
            {{GrammarTestHarness.Preamble}}

            namespace Fixtures
            {
                [Versionable(Version = 2, Hash = "5e2abc", OldNames = new[] { "OldConfig" })]
                [SerializationName("Config")]
                public sealed partial class Config
                {
                    public string Name { get; init; } = "";

                    public int Retries { get; set; }

                    [VersionableField("tag_list")]
                    public List<string> Tags { get; init; } = new();
                }
            }
            """, out Assembly assembly);

        Assert.Equal("Config", Read<string>(metadata, "Name"));
        Assert.Equal(2, Read<int>(metadata, "Version"));
        Assert.Equal("5e2abc", Read<string>(metadata, "Hash"));
        Assert.Equal(new[] { "OldConfig" }, Read<IEnumerable<string>>(metadata, "OldNames"));

        IReadOnlyList<object> fields = Fields(metadata);
        Assert.Equal(
            new[] { "Name", "Retries", "tag_list" },
            fields.Select(field => Read<string>(field, "WireName")));
        Assert.Equal(
            new[] { "str", "int", "list[str]" },
            fields.Select(field => Read<string>(field, "CanonicalType")));

        // Declaration order is what the factory expects, and the factory is what init-only
        // and positional members are rebuilt through.
        Func<object?[], object> factory = Read<Func<object?[], object>>(metadata, "Factory");
        object instance = factory(new object?[] { "prod", 3, new List<string> { "a", "b" } });

        Assert.Equal("prod", Read<Func<object, object?>>(fields[0], "Getter")(instance));
        Assert.Equal(3, Read<Func<object, object?>>(fields[1], "Getter")(instance));
        Assert.Equal(new[] { "a", "b" }, (IEnumerable<string>)Read<Func<object, object?>>(fields[2], "Getter")(instance)!);

        // A plain `set` accessor gets a setter; `init` members get null and must go through
        // the factory (FieldDescriptor contract).
        Assert.Null(ReadOrNull(fields[0], "Setter"));
        Action<object, object?> setter = Read<Action<object, object?>>(fields[1], "Setter");
        setter(instance, 9);
        Assert.Equal(9, Read<Func<object, object?>>(fields[1], "Getter")(instance));

        Assert.Same(metadata, RegisteredByName(assembly, "Config"));
        Assert.Same(metadata, RegisteredByName(assembly, "OldConfig"));
    }

    [Fact]
    public void a_register_false_type_is_reachable_by_clr_type_but_claims_no_name()
    {
        object metadata = Metadata("RuntimeHidden", $$"""
            {{GrammarTestHarness.Preamble}}

            namespace Fixtures
            {
                [Versionable(Version = 1, Hash = "6d52fd", Register = false)]
                [SerializationName("Hidden")]
                public sealed partial class Hidden
                {
                    public int Value { get; init; }
                }
            }
            """, "Fixtures.Hidden", out Assembly assembly);

        Assert.Null(RegisteredByName(assembly, "Hidden"));
        Assert.Same(metadata, RegisteredByType(assembly, Read<Type>(metadata, "ClrType")));
    }

    [Fact]
    public void every_generated_container_shape_materializes_from_the_wire()
    {
        // One fixture, one load context, every shape the engine refuses to build on its own.
        // Elements go back through WireValues, so an enum arrives as its [EnumValue] string and
        // a nested object as its envelope — the generator only builds the container.
        object metadata = Metadata("RuntimeShapes", $$"""
            {{GrammarTestHarness.Preamble}}

            namespace Fixtures
            {
                [SerializationName("Shade")]
                public enum Shade
                {
                    [EnumValue("red")] Red,
                    [EnumValue("blue")] Blue,
                }

                [Versionable(Version = 1, Hash = "34484d")]
                [SerializationName("Leaf")]
                public sealed partial class Leaf
                {
                    public int Weight { get; init; }
                }

                [Versionable(Version = 1, Hash = "5cc45c")]
                public sealed partial class Shapes
                {
                    public List<int> Counts { get; init; } = new();

                    public string[] Names { get; init; } = [];

                    public HashSet<string> Tags { get; init; } = new();

                    public FrozenSet<int> Ids { get; init; } = null!;

                    public Dictionary<string, double> Table { get; init; } = new();

                    public Dictionary<Shade, int> ByShade { get; init; } = new();

                    public (int, string) Row { get; init; }

                    public List<Shade> Palette { get; init; } = new();

                    public List<Leaf> Leaves { get; init; } = new();

                    public List<List<int>> Matrix { get; init; } = new();

                    public List<int>? Maybe { get; init; }
                }
            }
            """, "Fixtures.Shapes", out Assembly assembly);

        Dictionary<string, Func<object?, object?>> readers = Fields(metadata).ToDictionary(
            field => Read<string>(field, "WireName"),
            field => Read<Func<object?, object?>>(field, "WireReader"),
            StringComparer.Ordinal);

        // Widths are erased on the wire, so a long has to land in an int field.
        Assert.Equal(new List<int> { 1, 2 }, readers["Counts"](new object?[] { 1L, 2L }));
        Assert.Equal(new[] { "a", "b" }, readers["Names"](new object?[] { "a", "b" }));
        Assert.Equal(
            new[] { "x", "y" },
            ((IEnumerable<string>)readers["Tags"](new object?[] { "y", "x" })!).OrderBy(tag => tag, StringComparer.Ordinal));
        Assert.Equal(new[] { 7 }, ((IEnumerable<int>)readers["Ids"](new object?[] { 7L })!).ToArray());
        Assert.Equal(
            new Dictionary<string, double> { ["k"] = 1.5 },
            readers["Table"](new Dictionary<string, object?> { ["k"] = 1.5 }));
        Assert.Equal((3, "s"), readers["Row"](new object?[] { 3L, "s" }));
        Assert.Equal(new List<List<int>> { new() { 1 } }, readers["Matrix"](new object?[] { new object?[] { 1L } }));

        // A null container stays null: the engine hands the reader every value it reads.
        Assert.Null(readers["Maybe"](null));
        Assert.Equal(new List<int> { 4 }, readers["Maybe"](new object?[] { 4L }));

        // Enums come back through EnumConverter, so the [EnumValue] string is what matches —
        // as a dictionary key too, which the writer stringified on the way out.
        Type shade = assembly.GetType("Fixtures.Shade")!;
        object red = Enum.Parse(shade, "Red");
        object blue = Enum.Parse(shade, "Blue");
        Assert.Equal(new[] { red, blue }, ((IEnumerable)readers["Palette"](new object?[] { "red", "blue" })!).Cast<object>());
        Assert.Equal(
            new[] { red },
            ((IEnumerable)readers["ByShade"](new Dictionary<string, object?> { ["red"] = 1L })!)
                .Cast<object>()
                .Select(entry => entry.GetType().GetProperty("Key")!.GetValue(entry)!));

        // A nested [Versionable] element is materialized straight from its own generated
        // metadata — no registry lookup, which is the AOT-safe path ADR-0003 asks for.
        object leaves = readers["Leaves"](new object?[]
        {
            new Dictionary<string, object?> { ["Weight"] = 5L },
        })!;
        object leaf = ((IEnumerable)leaves).Cast<object>().Single();
        Assert.Equal("Fixtures.Leaf", leaf.GetType().FullName);
        Assert.Equal(5, leaf.GetType().GetProperty("Weight")!.GetValue(leaf));
    }

    [Fact]
    public void a_nullable_value_type_container_survives_being_nested()
    {
        // Regression: these three shapes rendered fine and raised no diagnostic, but emitted an
        // untyped `cond ? null : x` in a position with no target type, so the failure landed as
        // a type-inference error in a consumer's build. `list[tuple[int, str] | None]` is an
        // ordinary Python shape, so the C# mirror has to hold it.
        object metadata = Metadata("RuntimeNullableShapes", $$"""
            {{GrammarTestHarness.Preamble}}

            namespace Fixtures
            {
                [Versionable(Version = 1, Hash = "26cf95")]
                public sealed partial class NullableShapes
                {
                    public List<(int, string)?> Rows { get; init; } = new();

                    public Dictionary<string, (int, int)?> Pairs { get; init; } = new();

                    public (int, (int, string)?) Nested { get; init; }
                }
            }
            """, "Fixtures.NullableShapes", out _);

        Dictionary<string, Func<object?, object?>> readers = Fields(metadata).ToDictionary(
            field => Read<string>(field, "WireName"),
            field => Read<Func<object?, object?>>(field, "WireReader"),
            StringComparer.Ordinal);

        Assert.Equal(
            new (int, string)?[] { (1, "a"), null },
            (IEnumerable<(int, string)?>)readers["Rows"](new object?[] { new object?[] { 1L, "a" }, null })!);

        Assert.Equal(
            new Dictionary<string, (int, int)?> { ["here"] = (2, 3), ["gone"] = null },
            (IReadOnlyDictionary<string, (int, int)?>)readers["Pairs"](new Dictionary<string, object?>
            {
                ["here"] = new object?[] { 2L, 3L },
                ["gone"] = null,
            })!);

        (int, (int, string)?) present = (4, (5, "b"));
        (int, (int, string)?) absent = (4, null);
        Assert.Equal(present, readers["Nested"](new object?[] { 4L, new object?[] { 5L, "b" } }));
        Assert.Equal(absent, readers["Nested"](new object?[] { 4L, null }));
    }

    [Fact]
    public void a_null_element_in_a_value_type_container_names_the_type_it_expected()
    {
        object metadata = Metadata("RuntimeNullElement", $$"""
            {{GrammarTestHarness.Preamble}}

            namespace Fixtures
            {
                [Versionable(Version = 1, Hash = "514f02")]
                public sealed partial class Counters
                {
                    public List<int> Counts { get; init; } = new();
                }
            }
            """, "Fixtures.Counters", out _);

        Func<object?, object?> reader = Read<Func<object?, object?>>(Fields(metadata)[0], "WireReader");

        // Compared by name, not by type: the fixture's load context has its own copy of
        // Versionable.dll, so its ConverterException is a different type identity from the test
        // host's — which is the whole point of the isolation.
        Exception error = Assert.ThrowsAny<Exception>(() => reader(new object?[] { 1L, null }));
        Assert.Equal("Versionable.Errors.ConverterException", error.GetType().FullName);
        Assert.Contains("'int'", error.Message, StringComparison.Ordinal);
    }

    private static object Metadata(string assemblyName, string source, out Assembly assembly) =>
        Metadata(assemblyName, source, "Fixtures.Config", out assembly);

    private static object Metadata(
        string assemblyName,
        string source,
        string typeName,
        out Assembly assembly)
    {
        (Compilation compilation, _) = GrammarTestHarness.Generate(source, assemblyName);
        Assert.Empty(compilation.GetDiagnostics().Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));

        using MemoryStream image = new();
        EmitResult result = compilation.Emit(image);
        Assert.True(result.Success, string.Join("\n", result.Diagnostics));

        image.Position = 0;
        string runtimeDirectory = Path.GetDirectoryName(typeof(VersionableAttribute).Assembly.Location)!;
        assembly = new FixtureLoadContext(runtimeDirectory).LoadFromStream(image);

        // Module initializers run on first access to the module, which reflection alone does
        // not guarantee; force it so registration is observable.
        RuntimeHelpers.RunModuleConstructor(assembly.ManifestModule.ModuleHandle);

        Type fixture = assembly.GetType(typeName)
            ?? throw new InvalidOperationException($"Fixture type '{typeName}' is not in the emitted assembly.");

        return fixture.GetProperty("VersionableMetadata", BindingFlags.Public | BindingFlags.Static)!
            .GetValue(null)!;
    }

    private static IReadOnlyList<object> Fields(object metadata) =>
        ((System.Collections.IEnumerable)Read<object>(metadata, "Fields")).Cast<object>().ToList();

    private static T Read<T>(object instance, string property) => (T)ReadOrNull(instance, property)!;

    private static object? ReadOrNull(object instance, string property) =>
        instance.GetType().GetProperty(property)!.GetValue(instance);

    private static object? RegisteredByName(Assembly fixture, string name) =>
        InvokeRegistry(fixture, "TryGetByName", name);

    private static object? RegisteredByType(Assembly fixture, Type clrType) =>
        InvokeRegistry(fixture, "TryGetByType", clrType);

    private static object? InvokeRegistry(Assembly fixture, string method, object key)
    {
        object?[] arguments = { key, null };

        return (bool)RegistryType(fixture).GetMethod(method)!.Invoke(null, arguments)! ? arguments[1] : null;
    }

    private static Type RegistryType(Assembly fixture) =>
        fixture.GetReferencedAssemblies()
            .Where(reference => reference.Name == "Versionable")
            .Select(reference => AssemblyLoadContext.GetLoadContext(fixture)!.LoadFromAssemblyName(reference))
            .Select(loaded => loaded.GetType("Versionable.VersionableRegistry")!)
            .Single();

    /// <summary>Loads the fixture and its private copy of the versionable runtime.</summary>
    private sealed class FixtureLoadContext(string probePath) : AssemblyLoadContext(isCollectible: false)
    {
        protected override Assembly? Load(AssemblyName assemblyName)
        {
            // Only the versionable runtime is duplicated: everything else falls through to the
            // default context, so BCL types stay shared and delegates cross the boundary.
            if (assemblyName.Name != "Versionable")
            {
                return null;
            }

            string candidate = Path.Combine(probePath, assemblyName.Name + ".dll");
            return File.Exists(candidate) ? LoadFromAssemblyPath(candidate) : null;
        }
    }
}
