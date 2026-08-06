using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Versionable.Analyzers.Grammar;

namespace Versionable.Analyzers;

/// <summary>
/// Validates the declared schema hash of every <c>[Versionable]</c> type against the hash
/// computed from the compiler's symbol model, and rejects the constructs the canonical type
/// grammar cannot express.
/// </summary>
/// <remarks>
/// Compile-time counterpart of Python's definition-time tripwire in
/// <c>Versionable.__init_subclass__</c> (<c>src/versionable/_base.py</c>), which raises
/// <c>HashMismatchError</c> at import time. Per ADR-0003 the C# check is a build error
/// instead — stronger, because a mismatch never reaches a running process.
/// <para>
/// Per-type checks run as symbol actions. Serialization Name uniqueness (GRAMMAR §9) cannot:
/// it is a property of the whole compilation, so claims are accumulated during the symbol
/// actions and adjudicated once at compilation end.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class SchemaHashAnalyzer : DiagnosticAnalyzer
{
    /// <inheritdoc />
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => VersionableDiagnostics.All;

    /// <inheritdoc />
    public override void Initialize(AnalysisContext context)
    {
        context.EnableConcurrentExecution();
        context.ConfigureGeneratedCodeAnalysis(
            GeneratedCodeAnalysisFlags.None);
        context.RegisterCompilationStartAction(Start);
    }

    private static void Start(CompilationStartAnalysisContext context)
    {
        if (context.Compilation.GetTypeByMetadataName(SchemaModelBuilder.VersionableAttributeName) is null)
        {
            // The runtime library is not referenced, so nothing in this compilation can be
            // Versionable. Bail before walking a single symbol.
            return;
        }

        NameClaims claims = new();

        context.RegisterSymbolAction(
            symbolContext => AnalyzeType(symbolContext, claims),
            SymbolKind.NamedType);

        context.RegisterCompilationEndAction(claims.Report);
    }

    private static void AnalyzeType(SymbolAnalysisContext context, NameClaims claims)
    {
        if (context.Symbol is not INamedTypeSymbol type)
        {
            return;
        }

        AttributeData? attribute = SchemaModelBuilder.FindVersionableAttribute(type);
        if (attribute is null)
        {
            return;
        }

        SchemaModel model = SchemaModelBuilder.Build(type, attribute);

        foreach (SchemaProblem problem in model.Problems)
        {
            context.ReportDiagnostic(Diagnostic.Create(
                VersionableDiagnostics.ById(problem.Id),
                problem.Location,
                problem.MessageArguments));
        }

        claims.Claim(model.SerializationName, type);
        foreach (INamedTypeSymbol referenced in model.ReferencedTypes)
        {
            claims.Claim(CanonicalTypeRenderer.SerializationName(referenced), referenced);
        }
    }

    /// <summary>
    /// Serialization Names claimed across the compilation, and by which types.
    /// </summary>
    /// <remarks>
    /// Concurrent because <see cref="AnalysisContext.EnableConcurrentExecution"/> lets symbol
    /// actions run in parallel; the compilation-end action is the only reader and runs after
    /// all of them.
    /// </remarks>
    private sealed class NameClaims
    {
        private readonly ConcurrentDictionary<string, ConcurrentDictionary<INamedTypeSymbol, byte>> _claims =
            new(System.StringComparer.Ordinal);

        internal void Claim(string name, INamedTypeSymbol type)
        {
            ConcurrentDictionary<INamedTypeSymbol, byte> claimants = _claims.GetOrAdd(
                name,
                _ => new ConcurrentDictionary<INamedTypeSymbol, byte>(SymbolEqualityComparer.Default));
            claimants[type.OriginalDefinition] = 0;
        }

        internal void Report(CompilationAnalysisContext context)
        {
            foreach (KeyValuePair<string, ConcurrentDictionary<INamedTypeSymbol, byte>> claim in _claims)
            {
                if (claim.Value.Count < 2)
                {
                    continue;
                }

                // Deterministic ordering so the message reads the same on every build,
                // whatever order the parallel symbol actions ran in.
                List<INamedTypeSymbol> claimants = claim.Value.Keys
                    .OrderBy(type => type.ToDisplayString(), System.StringComparer.Ordinal)
                    .ToList();

                foreach (INamedTypeSymbol claimant in claimants)
                {
                    context.ReportDiagnostic(Diagnostic.Create(
                        VersionableDiagnostics.DuplicateSerializationNameRule,
                        claimant.Locations.FirstOrDefault() ?? Location.None,
                        claim.Key,
                        claimants[0].ToDisplayString(),
                        claimants[1].ToDisplayString()));
                }
            }
        }
    }
}
