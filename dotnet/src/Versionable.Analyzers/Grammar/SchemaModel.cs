using System.Collections.Immutable;
using Microsoft.CodeAnalysis;

namespace Versionable.Analyzers.Grammar;

/// <summary>A construct that cannot be expressed, paired with where to report it.</summary>
/// <remarks>
/// These build-time models use settable properties rather than <c>required</c>/<c>init</c>
/// members: the analyzer targets netstandard2.0, where both features need
/// <c>IsExternalInit</c> / <c>RequiredMemberAttribute</c> polyfills declared in
/// <c>System.Runtime.CompilerServices</c> — a namespace this repo's
/// <c>dotnet_style_namespace_match_folder</c> rule forbids outside a matching folder.
/// </remarks>
internal sealed class SchemaProblem
{
    /// <summary>Initializes a new instance of the <see cref="SchemaProblem"/> class.</summary>
    /// <param name="id">The <c>VSN</c> diagnostic id.</param>
    /// <param name="location">Where to report it.</param>
    /// <param name="messageArguments">Arguments for the diagnostic's message format.</param>
    internal SchemaProblem(string id, Location location, params object?[] messageArguments)
    {
        Id = id;
        Location = location;
        MessageArguments = messageArguments;
    }

    /// <summary>The <c>VSN</c> diagnostic id.</summary>
    internal string Id { get; }

    /// <summary>Where to report it.</summary>
    internal Location Location { get; }

    /// <summary>Arguments for the diagnostic's message format.</summary>
    internal object?[] MessageArguments { get; }
}

/// <summary>One serializable member of a <c>[Versionable]</c> type, resolved from symbols.</summary>
internal sealed class SchemaField
{
    /// <summary>The declaring property or field symbol.</summary>
    internal ISymbol Member { get; set; } = null!;

    /// <summary>Key written to and read from the file. Hash-significant.</summary>
    internal string WireName { get; set; } = string.Empty;

    /// <summary>Declaring property or field name in C#.</summary>
    internal string ClrName { get; set; } = string.Empty;

    /// <summary>Declared CLR type of the member.</summary>
    internal ITypeSymbol ClrType { get; set; } = null!;

    /// <summary>Canonical grammar rendering of <see cref="ClrType"/>.</summary>
    internal string CanonicalType { get; set; } = string.Empty;

    /// <summary>Whether an object initializer can assign the member (a <c>set</c> or <c>init</c> accessor).</summary>
    internal bool IsInitializable { get; set; }

    /// <summary>Whether a post-construction setter delegate can be emitted (a plain <c>set</c> accessor).</summary>
    internal bool IsAssignable { get; set; }

    /// <summary>The <c>[LiteralValues]</c> options in canonical order, or empty when not a literal field.</summary>
    internal ImmutableArray<TypedConstant> LiteralValues { get; set; } = ImmutableArray<TypedConstant>.Empty;

    /// <summary>Whether the member carries <c>[LiteralValues]</c>.</summary>
    internal bool IsLiteral { get; set; }

    /// <summary>Whether <c>[LiteralValues(Fallback = ...)]</c> was declared.</summary>
    internal bool HasLiteralFallback { get; set; }

    /// <summary>The declared fallback, when <see cref="HasLiteralFallback"/> is set.</summary>
    internal TypedConstant LiteralFallback { get; set; }

    /// <summary>The C# source of a literal default initializer, or <see langword="null"/>.</summary>
    internal string? DefaultExpression { get; set; }

    /// <summary>Where to report diagnostics about this member.</summary>
    internal Location Location { get; set; } = Location.None;
}

/// <summary>One migration declared in a type's nested <c>Migrate</c> class.</summary>
/// <remarks>
/// The member name is what the generator emits a reference to, so the pair (version, member) is
/// the whole of what a compile-time chain needs: <c>MigrationStep.Of(1, Migrate.V1)</c> for a
/// declarative member, and the same with a method group for an imperative one.
/// </remarks>
internal sealed class SchemaMigration
{
    /// <summary>Initializes a new instance of the <see cref="SchemaMigration"/> class.</summary>
    /// <param name="fromVersion">Schema version the migration reads.</param>
    /// <param name="memberName">Name of the member on the <c>Migrate</c> class.</param>
    /// <param name="imperative">Whether the member is a <c>[Migration]</c> method.</param>
    internal SchemaMigration(int fromVersion, string memberName, bool imperative)
    {
        FromVersion = fromVersion;
        MemberName = memberName;
        Imperative = imperative;
    }

    /// <summary>Schema version the migration reads; it produces data at one version later.</summary>
    internal int FromVersion { get; }

    /// <summary>Name of the member on the <c>Migrate</c> class.</summary>
    internal string MemberName { get; }

    /// <summary>Whether the member is a <c>[Migration]</c> method rather than a builder member.</summary>
    internal bool Imperative { get; }
}

/// <summary>How the generator reconstructs an instance from field values.</summary>
internal sealed class FactoryPlan
{
    /// <summary>The constructor to call.</summary>
    internal IMethodSymbol Constructor { get; set; } = null!;

    /// <summary>Field index per constructor parameter, positionally.</summary>
    internal ImmutableArray<int> ParameterFields { get; set; } = ImmutableArray<int>.Empty;

    /// <summary>Field indices assigned through an object initializer after construction.</summary>
    internal ImmutableArray<int> InitializerFields { get; set; } = ImmutableArray<int>.Empty;
}

/// <summary>
/// Everything the analyzer and the source generator need about one <c>[Versionable]</c>
/// type, resolved once from the symbol model.
/// </summary>
internal sealed class SchemaModel
{
    /// <summary>The annotated type.</summary>
    internal INamedTypeSymbol Type { get; set; } = null!;

    /// <summary>Serialization Name: <c>[SerializationName]</c> when declared, else the bare type name.</summary>
    internal string SerializationName { get; set; } = string.Empty;

    /// <summary>Declared <c>Version</c>.</summary>
    internal int Version { get; set; }

    /// <summary>Declared <c>Hash</c>, possibly empty.</summary>
    internal string DeclaredHash { get; set; } = string.Empty;

    /// <summary>Declared <c>OldNames</c>.</summary>
    internal ImmutableArray<string> OldNames { get; set; } = ImmutableArray<string>.Empty;

    /// <summary>Declared <c>Register</c>.</summary>
    internal bool Register { get; set; } = true;

    /// <summary>Declared <c>SkipDefaults</c>.</summary>
    internal bool SkipDefaults { get; set; }

    /// <summary>Declared <c>ValidateLiterals</c>.</summary>
    internal bool ValidateLiterals { get; set; } = true;

    /// <summary>Declared <c>Unknown</c>, as an <c>UnknownFieldPolicy</c> member name.</summary>
    internal string UnknownPolicy { get; set; } = "Ignore";

    /// <summary>Serializable members in declaration order.</summary>
    internal ImmutableArray<SchemaField> Fields { get; set; } = ImmutableArray<SchemaField>.Empty;

    /// <summary>The canonical payload these fields hash to.</summary>
    internal string Payload { get; set; } = string.Empty;

    /// <summary>The hash computed from <see cref="Payload"/>.</summary>
    internal string ComputedHash { get; set; } = string.Empty;

    /// <summary>Constructs that could not be expressed, with locations.</summary>
    internal ImmutableArray<SchemaProblem> Problems { get; set; } = ImmutableArray<SchemaProblem>.Empty;

    /// <summary>Named types reachable from the fields that claim a Serialization Name.</summary>
    internal ImmutableArray<INamedTypeSymbol> ReferencedTypes { get; set; } = ImmutableArray<INamedTypeSymbol>.Empty;

    /// <summary>
    /// Migrations declared as members of the nested <c>Migrate</c> class, ascending by source
    /// version. Empty when <see cref="MigrateTypeIsChain"/> is set, which supersedes them.
    /// </summary>
    internal ImmutableArray<SchemaMigration> Migrations { get; set; } = ImmutableArray<SchemaMigration>.Empty;

    /// <summary>
    /// Whether the nested <c>Migrate</c> type is itself an instantiable
    /// <c>IMigrationChain</c>, which the generator hands straight to
    /// <c>VersionableMetadata.Migrations</c>.
    /// </summary>
    /// <remarks>
    /// The escape hatch for a chain assembled at run time. Its source versions are a run-time
    /// value, so neither the contiguity check nor the member collection applies to it — see
    /// <c>VersionableMetadataGenerator.RenderBody</c>.
    /// </remarks>
    internal bool MigrateTypeIsChain { get; set; }

    /// <summary>Whether the type and every containing type is declared <c>partial</c>.</summary>
    internal bool IsPartial { get; set; }

    /// <summary>
    /// Whether a base type already carries the generated members, so this type's copies have to
    /// be declared <c>new</c>.
    /// </summary>
    /// <remarks>
    /// The polymorphic case GRAMMAR §9 blesses: <c>GoldenCircle : GoldenShape</c> where both are
    /// <c>[Versionable]</c>. Each needs its own metadata — different fields, different hash,
    /// different envelope name — and the derived declarations hide the base's, which is CS0108
    /// and therefore a build error under the <c>TreatWarningsAsErrors</c> this repo and its
    /// consumers use.
    /// </remarks>
    internal bool HidesBaseMembers { get; set; }

    /// <summary>
    /// Whether a <c>[ModuleInitializer]</c> can be emitted for the type: module-level
    /// accessibility, no generic enclosing types.
    /// </summary>
    internal bool SupportsModuleInitializer { get; set; }

    /// <summary>How to reconstruct an instance, or <see langword="null"/> when no constructor fits.</summary>
    internal FactoryPlan? Factory { get; set; }

    /// <summary>
    /// Whether the type is abstract, and so is a polymorphic base rather than something a load
    /// ever constructs directly.
    /// </summary>
    /// <remarks>
    /// It still needs metadata: a field declared as the base resolves its concrete type from
    /// the value's own envelope, and the base's metadata is the upper bound that resolution
    /// checks against. What it does not need is a factory — <c>new</c> on an abstract type does
    /// not compile — so the generator emits one that explains itself instead.
    /// </remarks>
    internal bool IsAbstract { get; set; }

    /// <summary>Whether the model is complete enough for the generator to emit code.</summary>
    internal bool CanGenerate => IsPartial && (IsAbstract || Factory is not null) && !HasBlockingProblem;

    private bool HasBlockingProblem
    {
        get
        {
            foreach (SchemaProblem problem in Problems)
            {
                // A wrong hash, a migration gap, and a disabled nullable context are all
                // reported without stopping generation: the emitted metadata is still
                // coherent, and failing to generate would bury the real diagnostic under a
                // cascade of "does not implement IVersionableMetadataProvider" errors.
                if (problem.Id != VersionableDiagnostics.HashMismatch
                    && problem.Id != VersionableDiagnostics.MigrationChainInvalid
                    && problem.Id != VersionableDiagnostics.NullableContextDisabled)
                {
                    return true;
                }
            }

            return false;
        }
    }
}
