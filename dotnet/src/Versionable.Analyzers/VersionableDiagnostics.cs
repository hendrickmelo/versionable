using System.Collections.Immutable;
using Microsoft.CodeAnalysis;

namespace Versionable.Analyzers;

/// <summary>
/// Diagnostic identifiers and descriptors reported by the Versionable analyzer.
/// </summary>
/// <remarks>
/// The <c>VSN</c> prefix is the Versionable diagnostic namespace. Severities are
/// configurable per-id from <c>.editorconfig</c>; <see cref="HashMismatch"/> is the
/// analogue of Python's <c>versionable.ignoreHashErrors()</c> dev mode
/// (<c>src/versionable/_base.py</c>).
/// <para>
/// Every message that reports a hash also reports the canonical payload it was computed
/// from. A bare "expected a1b2c3, got d4e5f6" is unactionable — the payload is what tells
/// the author <em>which</em> field drifted (ADR-0003).
/// </para>
/// </remarks>
internal static class VersionableDiagnostics
{
    /// <summary>Category applied to every Versionable diagnostic.</summary>
    internal const string Category = "Versionable";

    /// <summary>Declared <c>Hash</c> disagrees with the hash computed from the symbol model.</summary>
    internal const string HashMismatch = "VSN0001";

    /// <summary>A declared type cannot be expressed in the canonical type grammar.</summary>
    internal const string UnsupportedType = "VSN0002";

    /// <summary>The migration chain has a gap above the oldest declared migration.</summary>
    internal const string MigrationChainGap = "VSN0003";

    /// <summary>Two types reachable from one schema claim the same Serialization Name.</summary>
    internal const string DuplicateSerializationName = "VSN0004";

    /// <summary>A <c>[Versionable]</c> type — or one of its enclosing types — is not <c>partial</c>.</summary>
    internal const string MustBePartial = "VSN0005";

    /// <summary>A <c>[LiteralValues]</c> option is of a kind the grammar's closed member list rejects.</summary>
    internal const string UnsupportedLiteralValue = "VSN0006";

    /// <summary>A wire name contains a payload separator or leaves the Basic Multilingual Plane.</summary>
    internal const string InvalidWireName = "VSN0007";

    /// <summary>Two members of one type claim the same wire name.</summary>
    internal const string DuplicateWireName = "VSN0008";

    /// <summary>A <c>[Versionable]</c> type is compiled without nullable reference types enabled.</summary>
    internal const string NullableContextDisabled = "VSN0009";

    /// <summary>No constructor can rebuild the type from its serializable members.</summary>
    internal const string NoUsableConstructor = "VSN0010";

    /// <summary>The type cannot carry a <c>[ModuleInitializer]</c>, so it never self-registers.</summary>
    internal const string NotSelfRegistering = "VSN0011";

    internal const string HelpLink =
        "https://github.com/hendrickmelo/versionable/blob/main/conformance/GRAMMAR.md";

    /// <summary>Declared <c>Hash</c> disagrees with the computed hash.</summary>
    internal static readonly DiagnosticDescriptor HashMismatchRule = Rule(
        HashMismatch,
        "Schema hash does not match the declared Hash",
        "Schema hash mismatch on '{0}': declared Hash = \"{1}\", computed \"{2}\" from payload '{3}'. "
            + "Update the declared hash to \"{2}\" if the schema change is intended.",
        "The hash computed from this type's fields disagrees with the Hash declared on [Versionable]. "
            + "Either the schema changed without the hash being updated, or the hash was mistyped. "
            + "The message carries the full canonical payload so the drifting field is identifiable.");

    /// <summary>A field's type has no canonical rendering.</summary>
    internal static readonly DiagnosticDescriptor UnsupportedTypeRule = Rule(
        UnsupportedType,
        "Type cannot be expressed in the canonical type grammar",
        "Field '{0}' of '{1}' cannot be hashed: {2}",
        "The canonical type grammar is a cross-language contract, so a construct with no grammar form is "
            + "rejected rather than rendered in a C#-specific way that no other implementation could reproduce.");

    /// <summary>The migration chain skips a version above its oldest entry.</summary>
    internal static readonly DiagnosticDescriptor MigrationChainGapRule = Rule(
        MigrationChainGap,
        "Migration chain is not contiguous",
        "Migration chain of '{0}' is not contiguous: {1}",
        "A load walks migrations one version at a time, so a gap above the oldest declared migration makes "
            + "every older file unreachable. Gaps below the oldest migration are legitimate — they are what "
            + "MinReversibleVersion reports. Scope limit: this check reads only the declarative members of "
            + "the nested Migrate class — static V1/V2/... fields or properties, and static methods carrying "
            + "[Migration(FromVersion = n)]. A Migrate type that is itself an IMigrationChain declares its "
            + "source versions in a FromVersions array evaluated at run time, which no compile-time check can "
            + "read, so an imperative chain's contiguity is not verified here. Phase 4 either extends this to "
            + "the builder-composed chain or documents the imperative form as unchecked.");

    /// <summary>Two types claim one Serialization Name.</summary>
    internal static readonly DiagnosticDescriptor DuplicateSerializationNameRule = Rule(
        DuplicateSerializationName,
        "Serialization Name is claimed by two types",
        "Serialization Name '{0}' is claimed by both '{1}' and '{2}'. Give one of them an explicit distinct "
            + "name with [SerializationName(\"...\")].",
        "The grammar drops namespaces, so two same-named types reachable from one schema hash identically "
            + "while meaning different things. Serialization Names must be unique across every reachable type. "
            + "This check spans the types declared in this compilation; a collision with a type in a referenced "
            + "assembly is caught at startup by VersionableRegistry.Register instead, because detecting it here "
            + "would mean walking every referenced assembly on every build.");

    /// <summary>A <c>[Versionable]</c> type is not <c>partial</c>.</summary>
    internal static readonly DiagnosticDescriptor MustBePartialRule = Rule(
        MustBePartial,
        "Versionable type must be partial",
        "'{0}' carries [Versionable] but {1} is not declared partial, so no metadata can be generated for it",
        "The generator implements IVersionableMetadataProvider's static abstract member by emitting a second "
            + "part of the type, which is only possible on a partial type. A non-partial [Versionable] type "
            + "drops out of the compile-time path entirely.");

    /// <summary>A <c>[LiteralValues]</c> option is of a rejected kind.</summary>
    internal static readonly DiagnosticDescriptor UnsupportedLiteralValueRule = Rule(
        UnsupportedLiteralValue,
        "Literal option cannot be expressed in the canonical type grammar",
        "[LiteralValues] on '{0}' of '{1}' is invalid: {2}",
        "The grammar's Literal member list is closed — string, int, bool, and null. Anything else would have "
            + "to be rendered in a language-specific way and would produce a hash no other implementation "
            + "could reproduce.");

    /// <summary>A wire name breaks the payload's structure.</summary>
    internal static readonly DiagnosticDescriptor InvalidWireNameRule = Rule(
        InvalidWireName,
        "Wire name is not valid in a hash payload",
        "Wire name '{0}' on '{1}' is invalid: {2}",
        "Wire names are joined into the hash payload with ':' and ',' and are re-splittable by bracket depth, "
            + "so ':' ',' '[' ']' are reserved. Grammar version 1 also restricts names to the Basic Multilingual "
            + "Plane so that C# ordinal ordering and Unicode code-point ordering agree.");

    /// <summary>Two members claim one wire name.</summary>
    internal static readonly DiagnosticDescriptor DuplicateWireNameRule = Rule(
        DuplicateWireName,
        "Wire name is claimed by two members",
        "Wire name '{0}' on '{1}' is claimed by more than one member",
        "The payload is a mapping from wire name to canonical type, and a file is a mapping from wire name to "
            + "value. Two members sharing a name make both ambiguous.");

    /// <summary>Nullable reference types are disabled for this type.</summary>
    internal static readonly DiagnosticDescriptor NullableContextDisabledRule = Rule(
        NullableContextDisabled,
        "Versionable type is compiled without nullable reference types",
        "'{0}' is compiled with nullable reference types disabled, so reference-typed fields cannot be told "
            + "apart from their nullable forms and hash as non-null",
        "The grammar renders T? as Union[None, T]. Without <Nullable>enable</Nullable> the compiler reports no "
            + "annotation at all, so a field meant to be optional silently hashes as required and diverges from "
            + "the Python schema it mirrors.",
        DiagnosticSeverity.Warning);

    /// <summary>No constructor can rebuild the type.</summary>
    internal static readonly DiagnosticDescriptor NoUsableConstructorRule = Rule(
        NoUsableConstructor,
        "Versionable type cannot be reconstructed on load",
        "'{0}' has no constructor that can rebuild it: {1}",
        "Loading builds an instance from field values through the generated Factory. Every member without a "
            + "settable accessor must therefore be reachable through a constructor parameter of the same name.");

    /// <summary>The type cannot self-register.</summary>
    internal static readonly DiagnosticDescriptor NotSelfRegisteringRule = Rule(
        NotSelfRegistering,
        "Versionable type cannot register itself",
        "'{0}' cannot carry a module initializer because {1}, so it is absent from VersionableRegistry and "
            + "cannot be the target of a polymorphic load. Its metadata is still reachable through "
            + "IVersionableMetadataProvider.",
        "Registration is emitted as a [ModuleInitializer], which the CLR only accepts on a method reachable "
            + "from module scope and outside any generic type. A type that fails those conditions still has "
            + "generated metadata, but nothing puts it in the name and type indexes.",
        DiagnosticSeverity.Warning);

    /// <summary>Every descriptor, for <c>SupportedDiagnostics</c>.</summary>
    internal static readonly ImmutableArray<DiagnosticDescriptor> All = ImmutableArray.Create(
        HashMismatchRule,
        UnsupportedTypeRule,
        MigrationChainGapRule,
        DuplicateSerializationNameRule,
        MustBePartialRule,
        UnsupportedLiteralValueRule,
        InvalidWireNameRule,
        DuplicateWireNameRule,
        NullableContextDisabledRule,
        NoUsableConstructorRule,
        NotSelfRegisteringRule);

    /// <summary>Finds a descriptor by id.</summary>
    /// <param name="id">A <c>VSN</c> id.</param>
    /// <returns>The matching descriptor.</returns>
    internal static DiagnosticDescriptor ById(string id)
    {
        foreach (DiagnosticDescriptor descriptor in All)
        {
            if (descriptor.Id == id)
            {
                return descriptor;
            }
        }

        return UnsupportedTypeRule;
    }

    private static DiagnosticDescriptor Rule(
        string id,
        string title,
        string messageFormat,
        string description,
        DiagnosticSeverity severity = DiagnosticSeverity.Error) =>
        new(
            id,
            title,
            messageFormat,
            Category,
            severity,
            isEnabledByDefault: true,
            description: description,
            helpLinkUri: HelpLink);
}
