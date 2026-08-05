using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Versionable.Analyzers;

/// <summary>
/// Validates the declared schema hash of every <c>[Versionable]</c> type against the hash
/// computed from the compiler's symbol model.
/// </summary>
/// <remarks>
/// Compile-time counterpart of Python's definition-time tripwire in
/// <c>Versionable.__init_subclass__</c> (<c>src/versionable/_base.py</c>), which raises
/// <c>HashMismatchError</c> at import time. Per ADR-0003 the C# check is a build error
/// instead. Stub: rules and analysis land in task 2b.
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class SchemaHashAnalyzer : DiagnosticAnalyzer
{
    /// <inheritdoc />
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
        ImmutableArray<DiagnosticDescriptor>.Empty;

    /// <inheritdoc />
    public override void Initialize(AnalysisContext context)
    {
        context.EnableConcurrentExecution();
        context.ConfigureGeneratedCodeAnalysis(
            GeneratedCodeAnalysisFlags.None);
    }
}
