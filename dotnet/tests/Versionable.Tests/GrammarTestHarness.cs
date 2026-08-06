using System.Collections.Immutable;
using System.Reflection;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Versionable.Analyzers;
using Versionable.Analyzers.Grammar;

namespace Versionable.Tests;

/// <summary>
/// Compiles fixture source in memory and runs the real analyzer, generator, and canonical
/// renderer over it.
/// </summary>
/// <remarks>
/// References come from the test host's own trusted-platform-assembly list, so the fixture
/// compilation sees exactly the assemblies this test run loaded — including
/// <c>Versionable.dll</c> and <c>System.Numerics.Tensors.dll</c>. That keeps the harness free
/// of reference-assembly packages, which would otherwise have to be kept in step with the
/// target framework by hand.
/// </remarks>
internal static class GrammarTestHarness
{
    /// <summary>Types the conformance fixtures refer to by Serialization Name.</summary>
    internal const string Preamble = """
        using System;
        using System.Collections.Frozen;
        using System.Collections.Generic;
        using System.ComponentModel;
        using System.Numerics;
        using System.Numerics.Tensors;
        using System.Text.RegularExpressions;
        using Versionable;

        public enum Status { Active, Idle }

        [SerializationName("DeviceStatus")]
        public enum LocalStatus { Active, Idle }

        public sealed class Node { }
        public sealed class Leaf { }
        public sealed class DeviceConfig { }
        public sealed class Calibration { }
        public sealed class ChannelConfig { }
        public sealed class MyBox<T> { }

        public sealed class Outer { public sealed class Inner { } }

        [SerializationName("Renamed")]
        public sealed class NeedsRenaming { }

        public abstract class AbstractThing { }

        [Versionable(Version = 1, Hash = "e3b0c4")]
        public abstract partial class AbstractShape { }

        [Versionable(Version = 1, Hash = "34484d")]
        public sealed partial class NestedLeaf
        {
            public int Weight { get; init; }
        }
        """;

    private static readonly ImmutableArray<MetadataReference> _references = LoadReferences();

    /// <summary>Compiles <paramref name="source"/> into an in-memory assembly.</summary>
    /// <param name="source">Complete C# source.</param>
    /// <param name="nullable">Nullable context for the compilation.</param>
    /// <param name="assemblyName">Assembly name, which must be unique per loadable test assembly.</param>
    /// <returns>The compilation.</returns>
    internal static CSharpCompilation Compile(
        string source,
        NullableContextOptions nullable = NullableContextOptions.Enable,
        string assemblyName = "VersionableFixtures")
    {
        return CSharpCompilation.Create(
            assemblyName,
            new[] { CSharpSyntaxTree.ParseText(source, new CSharpParseOptions(LanguageVersion.CSharp12)) },
            _references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: nullable));
    }

    /// <summary>Wraps fixture members in a <c>[Versionable]</c> type and resolves its schema.</summary>
    /// <param name="members">Member declarations for the fixture type.</param>
    /// <param name="declaredHash">Hash to declare on the attribute.</param>
    /// <returns>The resolved schema model.</returns>
    internal static SchemaModel Schema(string members, string declaredHash = "")
    {
        string source = $$"""
            {{Preamble}}

            [Versionable(Version = 1, Hash = "{{declaredHash}}")]
            public sealed partial class Fixture
            {
            {{members}}
            }
            """;

        return Schema(Compile(source), "Fixture");
    }

    /// <summary>Resolves the schema of a named <c>[Versionable]</c> type in a compilation.</summary>
    /// <param name="compilation">A compilation containing the type.</param>
    /// <param name="metadataName">The type's metadata name.</param>
    /// <returns>The resolved schema model.</returns>
    internal static SchemaModel Schema(CSharpCompilation compilation, string metadataName)
    {
        INamedTypeSymbol type = compilation.GetTypeByMetadataName(metadataName)
            ?? throw new InvalidOperationException($"Fixture type '{metadataName}' did not compile.");
        AttributeData attribute = SchemaModelBuilder.FindVersionableAttribute(type)
            ?? throw new InvalidOperationException($"Fixture type '{metadataName}' has no [Versionable].");

        return SchemaModelBuilder.Build(type, attribute);
    }

    /// <summary>Renders one declared type through the canonical grammar.</summary>
    /// <param name="declaration">A single member declaration, for example <c>public int x;</c>.</param>
    /// <returns>The canonical type string of that member.</returns>
    internal static string Render(string declaration) => Schema("    " + declaration).Fields[0].CanonicalType;

    /// <summary>Runs the analyzer over a compilation and returns only Versionable diagnostics.</summary>
    /// <param name="source">Complete C# source, including the preamble if it is needed.</param>
    /// <param name="nullable">Nullable context for the compilation.</param>
    /// <returns>The <c>VSN</c> diagnostics, ordered by id then message.</returns>
    internal static ImmutableArray<Diagnostic> Analyze(
        string source,
        NullableContextOptions nullable = NullableContextOptions.Enable)
    {
        CSharpCompilation compilation = Compile(source, nullable);
        CompilationWithAnalyzers analyzed = compilation.WithAnalyzers(
            ImmutableArray.Create<DiagnosticAnalyzer>(new SchemaHashAnalyzer()));

        return analyzed.GetAnalyzerDiagnosticsAsync().GetAwaiter().GetResult()
            .Where(diagnostic => diagnostic.Id.StartsWith("VSN", StringComparison.Ordinal))
            .OrderBy(diagnostic => diagnostic.Id, StringComparer.Ordinal)
            .ThenBy(diagnostic => diagnostic.GetMessage(), StringComparer.Ordinal)
            .ToImmutableArray();
    }

    /// <summary>Runs the source generator and returns the compilation it produced.</summary>
    /// <param name="source">Complete C# source.</param>
    /// <param name="assemblyName">Assembly name for the fixture compilation.</param>
    /// <returns>The post-generation compilation and every source the generator added.</returns>
    internal static (Compilation Compilation, IReadOnlyList<(string HintName, string Text)> Generated) Generate(
        string source,
        string assemblyName = "VersionableFixtures")
    {
        CSharpCompilation compilation = Compile(source, assemblyName: assemblyName);
        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            new[] { new VersionableMetadataGenerator().AsSourceGenerator() },
            parseOptions: new CSharpParseOptions(LanguageVersion.CSharp12));

        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out Compilation updated, out _);

        List<(string, string)> generated = driver.GetRunResult().Results
            .SelectMany(result => result.GeneratedSources)
            .Select(generatedSource => (generatedSource.HintName, generatedSource.SourceText.ToString()))
            .ToList();

        return (updated, generated);
    }

    private static ImmutableArray<MetadataReference> LoadReferences()
    {
        string platform = (string)(AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") ?? string.Empty);
        IEnumerable<string> paths = platform
            .Split(Path.PathSeparator)
            .Where(path => path.EndsWith(".dll", StringComparison.OrdinalIgnoreCase));

        // The versionable runtime is in the list only if this test host already loaded it, so
        // add it explicitly rather than relying on the deps graph.
        paths = paths.Concat(new[] { typeof(VersionableAttribute).Assembly.Location });

        return paths
            .Where(path => path.Length > 0 && File.Exists(path))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Select(path => (MetadataReference)MetadataReference.CreateFromFile(path))
            .ToImmutableArray();
    }

    /// <summary>Locates the repository root from the test assembly's location.</summary>
    /// <returns>The absolute repository root path.</returns>
    internal static string RepositoryRoot()
    {
        DirectoryInfo? directory = new(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "conformance", "GRAMMAR.md")))
        {
            directory = directory.Parent;
        }

        return directory?.FullName
            ?? throw new InvalidOperationException("Could not locate the repository root from the test assembly.");
    }
}
