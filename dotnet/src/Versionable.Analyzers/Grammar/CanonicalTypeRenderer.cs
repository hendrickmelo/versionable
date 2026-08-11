using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.Linq;
using Microsoft.CodeAnalysis;

namespace Versionable.Analyzers.Grammar;

/// <summary>
/// Renders a Roslyn <see cref="ITypeSymbol"/> into the canonical type string defined by
/// <c>conformance/GRAMMAR.md</c>. Shared by the analyzer and the source generator, which is
/// the point of ADR-0003: one rendering, two consumers, no chance of them disagreeing.
/// </summary>
/// <remarks>
/// Python counterpart: <c>canonicalTypeName()</c> in <c>src/versionable/_hash.py</c>. The
/// two implementations must agree byte for byte; <c>conformance/hash-vectors.json</c> is the
/// executable contract that says so.
/// <para>
/// Constructs the grammar cannot express are recorded in <see cref="Problems"/> rather than
/// thrown: the caller turns them into diagnostics with a source location, and rendering
/// continues so one bad field does not hide the rest.
/// </para>
/// </remarks>
internal sealed class CanonicalTypeRenderer
{
    private static readonly SymbolDisplayFormat _describeFormat = new(
        globalNamespaceStyle: SymbolDisplayGlobalNamespaceStyle.Omitted,
        typeQualificationStyle: SymbolDisplayTypeQualificationStyle.NameAndContainingTypesAndNamespaces,
        genericsOptions: SymbolDisplayGenericsOptions.IncludeTypeParameters,
        miscellaneousOptions: SymbolDisplayMiscellaneousOptions.UseSpecialTypes);

    private readonly List<string> _problems = new();

    /// <summary>Descriptions of constructs that could not be expressed in the grammar.</summary>
    internal IReadOnlyList<string> Problems => _problems;

    /// <summary>Whether every rendering so far was expressible.</summary>
    internal bool IsClean => _problems.Count == 0;

    /// <summary>
    /// Returns the Serialization Name of <paramref name="type"/>: the value of
    /// <c>[SerializationName]</c> when declared, otherwise the bare type name (§9).
    /// </summary>
    /// <param name="type">A named type.</param>
    /// <returns>A bare identifier — never namespace- or arity-qualified.</returns>
    internal static string SerializationName(INamedTypeSymbol type) =>
        DeclaredSerializationName(type) ?? type.OriginalDefinition.Name;

    /// <summary>
    /// The value of <c>[SerializationName]</c> on <paramref name="type"/>, or
    /// <see langword="null"/> when it declares none.
    /// </summary>
    /// <param name="type">A named type.</param>
    /// <returns>The declared override, or <see langword="null"/>.</returns>
    internal static string? DeclaredSerializationName(INamedTypeSymbol type)
    {
        foreach (AttributeData attribute in type.OriginalDefinition.GetAttributes())
        {
            if (attribute.AttributeClass is { Name: "SerializationNameAttribute" }
                && FullName(attribute.AttributeClass) == "Versionable.SerializationNameAttribute"
                && attribute.ConstructorArguments.Length == 1
                && attribute.ConstructorArguments[0].Value is string declared)
            {
                return declared;
            }
        }

        return null;
    }

    /// <summary>
    /// Fully-qualified, keyword-free, arity-free name of <paramref name="type"/> — for
    /// example <c>System.Collections.Generic.List</c>. Used only for table lookups.
    /// </summary>
    /// <param name="type">Any type symbol.</param>
    /// <returns>The namespace, containing type names, and type name, dot-joined.</returns>
    internal static string FullName(ITypeSymbol type)
    {
        ITypeSymbol definition = type.OriginalDefinition;
        List<string> parts = new();
        for (INamedTypeSymbol? current = definition as INamedTypeSymbol; current is not null; current = current.ContainingType)
        {
            parts.Insert(0, current.Name);
        }

        if (parts.Count == 0)
        {
            parts.Add(definition.Name);
        }

        INamespaceSymbol? containingNamespace = definition.ContainingNamespace;
        string prefix = containingNamespace is null || containingNamespace.IsGlobalNamespace
            ? string.Empty
            : containingNamespace.ToDisplayString() + ".";

        return prefix + string.Join(".", parts);
    }

    /// <summary>Renders <paramref name="type"/>, honoring nullability at every level.</summary>
    /// <param name="type">The declared type of a field or a type argument.</param>
    /// <returns>The canonical type string.</returns>
    internal string Render(ITypeSymbol type)
    {
        if (TryUnwrapNullable(type, out ITypeSymbol inner))
        {
            return CanonicalGrammar.RenderUnion(new[] { CanonicalGrammar.NoneToken, Render(inner) });
        }

        return RenderNonNullable(type);
    }

    /// <summary>
    /// Renders a field declared with <c>[LiteralValues(...)]</c>. Attribute argument order is
    /// the canonical order and is never sorted (§8); a nullable declared type still wraps the
    /// result in a <c>Union</c> with <c>None</c>.
    /// </summary>
    /// <param name="declaredType">The property or field's declared type.</param>
    /// <param name="values">The attribute's positional arguments, in declaration order.</param>
    /// <returns>The canonical type string.</returns>
    internal string RenderLiteral(ITypeSymbol declaredType, ImmutableArray<TypedConstant> values)
    {
        string literal = "Literal[" + string.Join(
            CanonicalGrammar.ArgumentSeparator,
            values.Select(RenderLiteralValue)) + "]";

        return TryUnwrapNullable(declaredType, out _)
            ? CanonicalGrammar.RenderUnion(new[] { CanonicalGrammar.NoneToken, literal })
            : literal;
    }

    /// <summary>
    /// Splits <paramref name="type"/> into "nullable of <paramref name="inner"/>" when it is
    /// a <c>Nullable&lt;T&gt;</c> or an annotated nullable reference type. The grammar treats
    /// the two identically (§6).
    /// </summary>
    /// <param name="type">The type to inspect.</param>
    /// <param name="inner">The non-null type, when <paramref name="type"/> is nullable.</param>
    /// <returns><see langword="true"/> when <paramref name="type"/> is nullable.</returns>
    internal static bool TryUnwrapNullable(ITypeSymbol type, out ITypeSymbol inner)
    {
        if (type is INamedTypeSymbol named
            && named.OriginalDefinition.SpecialType == SpecialType.System_Nullable_T
            && named.TypeArguments.Length == 1)
        {
            inner = named.TypeArguments[0];
            return true;
        }

        if (type.IsReferenceType && type.NullableAnnotation == NullableAnnotation.Annotated)
        {
            inner = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
            return true;
        }

        inner = type;
        return false;
    }

    private string RenderNonNullable(ITypeSymbol type)
    {
        if (type is IArrayTypeSymbol array)
        {
            if (array.Rank != 1)
            {
                Reject($"the multi-dimensional array '{Describe(type)}' has no canonical form; use Tensor<T> or a jagged array");
            }

            // byte[] is the carve-out: 'bytes', not list[int] (§7). byte?[] is not — its
            // element is Nullable<byte>, so it falls through to the container rendering.
            return array.ElementType.SpecialType == SpecialType.System_Byte && array.Rank == 1
                ? "bytes"
                : "list[" + Render(array.ElementType) + "]";
        }

        if (type is IDynamicTypeSymbol)
        {
            Reject("'dynamic' has no canonical form");
            return "Object";
        }

        if (type is ITypeParameterSymbol typeParameter)
        {
            Reject($"the type parameter '{typeParameter.Name}' has no canonical form; Versionable types cannot be generic");
            return typeParameter.Name;
        }

        if (type is not INamedTypeSymbol named)
        {
            Reject($"'{Describe(type)}' has no canonical form");
            return type.Name;
        }

        switch (named.SpecialType)
        {
            case SpecialType.System_Boolean:
                return "bool";
            case SpecialType.System_SByte:
            case SpecialType.System_Byte:
            case SpecialType.System_Int16:
            case SpecialType.System_UInt16:
            case SpecialType.System_Int32:
            case SpecialType.System_UInt32:
            case SpecialType.System_Int64:
            case SpecialType.System_UInt64:
            case SpecialType.System_IntPtr:
            case SpecialType.System_UIntPtr:
                return "int";
            case SpecialType.System_Single:
            case SpecialType.System_Double:
                return "float";
            case SpecialType.System_String:
            // A char is a one-character string on the wire; Python has no char type, so the
            // width-erasure principle of §4 applies to text as it does to numbers.
            case SpecialType.System_Char:
                return "str";
            case SpecialType.System_Decimal:
                return "Decimal";
            default:
                break;
        }

        if (TryRenderTuple(named, out string tuple))
        {
            return tuple;
        }

        if (named.TypeKind == TypeKind.Enum)
        {
            return SerializationName(named);
        }

        string full = FullName(named);
        if (CanonicalGrammar.ConverterNames.TryGetValue(full, out string? converterName)
            && !HasExplicitSerializationName(named))
        {
            return converterName;
        }

        switch (full)
        {
            case "System.Half":
                // Binary floating point of any width erases to `float` (§4). Only an array
                // dtype keeps its width.
                return "float";
            case "System.Numerics.Complex":
                return "complex";
            default:
                break;
        }

        if (named.Arity == 1 && named.TypeArguments.Length == 1)
        {
            switch (full)
            {
                case "System.Collections.Generic.List":
                    return "list[" + Render(named.TypeArguments[0]) + "]";
                case "System.Collections.Generic.HashSet":
                    return "set[" + Render(named.TypeArguments[0]) + "]";
                case "System.Collections.Frozen.FrozenSet":
                case "System.Collections.Immutable.ImmutableHashSet":
                    return "frozenset[" + Render(named.TypeArguments[0]) + "]";
                case "System.Numerics.Tensors.Tensor":
                    return RenderTensor(named.TypeArguments[0]);
                default:
                    break;
            }
        }

        if (full == "System.Collections.Generic.Dictionary" && named.TypeArguments.Length == 2)
        {
            return "dict[" + Render(named.TypeArguments[0])
                + CanonicalGrammar.ArgumentSeparator + Render(named.TypeArguments[1]) + "]";
        }

        // Everything else — Versionable types, enums, converter types, and unregistered types
        // alike — renders as a bare Serialization Name with any type parameters dropped (§9),
        // as long as it names something the loader could actually build.
        if (!IsRenderableAsName(named, out string reason))
        {
            Reject(reason);
        }

        return SerializationName(named);
    }

    private static bool HasExplicitSerializationName(INamedTypeSymbol type) =>
        DeclaredSerializationName(type) is not null;

    /// <summary>
    /// Rejects the types that would render as a bare Serialization Name the runtime could
    /// never honor.
    /// </summary>
    /// <remarks>
    /// GRAMMAR §9 lets an unrecognised type render its bare class name, and for a concrete
    /// class that is right — a converter can be registered for it later, and Python does the
    /// same. It is wrong for a type that can never <em>be</em> anything: <c>object</c> and
    /// <c>dynamic</c> name no schema at all, an interface or a non-Versionable abstract class
    /// names one the loader cannot construct, and a delegate has no wire form. Each of those
    /// would produce a hash that looks conformant and a file that cannot be read back, so the
    /// grammar's fall-through stops short of them. An abstract class that carries
    /// <c>[Versionable]</c> is fine — that is the polymorphic-base case, and the envelope
    /// names the concrete type.
    /// </remarks>
    private static bool IsRenderableAsName(INamedTypeSymbol type, out string reason)
    {
        reason = string.Empty;

        if (type.SpecialType == SpecialType.System_Object)
        {
            reason = "'object' names no schema; declare the concrete type, or a [Versionable] base type "
                + "for a polymorphic field";
            return false;
        }

        if (type.TypeKind == TypeKind.Interface)
        {
            reason = $"the interface '{Describe(type)}' names a schema the loader cannot construct; "
                + "declare the concrete type";
            return false;
        }

        if (type.TypeKind == TypeKind.Delegate)
        {
            reason = $"the delegate type '{Describe(type)}' has no wire form";
            return false;
        }

        if (type.IsAbstract && type.TypeKind == TypeKind.Class && !IsVersionable(type))
        {
            reason = $"the abstract class '{Describe(type)}' is not [Versionable], so nothing on the wire "
                + "can name a concrete type to build";
            return false;
        }

        return true;
    }

    private static bool IsVersionable(INamedTypeSymbol type)
    {
        foreach (AttributeData attribute in type.OriginalDefinition.GetAttributes())
        {
            if (attribute.AttributeClass is not null
                && FullName(attribute.AttributeClass) == "Versionable.VersionableAttribute")
            {
                return true;
            }
        }

        return false;
    }

    private bool TryRenderTuple(INamedTypeSymbol named, out string rendered)
    {
        if (named.IsTupleType)
        {
            rendered = "tuple[" + string.Join(
                CanonicalGrammar.ArgumentSeparator,
                named.TupleElements.Select(element => Render(element.Type))) + "]";
            return true;
        }

        // ValueTuple<T> of arity 1 has no tuple syntax, so Roslyn may not report it as a
        // tuple type; the grammar still spells it tuple[T].
        if (FullName(named) == "System.ValueTuple" && named.TypeArguments.Length > 0)
        {
            rendered = "tuple[" + string.Join(
                CanonicalGrammar.ArgumentSeparator,
                named.TypeArguments.Select(Render)) + "]";
            return true;
        }

        rendered = string.Empty;
        return false;
    }

    private string RenderTensor(ITypeSymbol elementType)
    {
        string full = FullName(elementType);
        if (!TryUnwrapNullable(elementType, out _)
            && CanonicalGrammar.DtypeTokens.TryGetValue(full, out string? token))
        {
            return "ndarray[" + token + "]";
        }

        Reject(
            $"'{Describe(elementType)}' is not one of the canonical ndarray dtypes "
            + $"({string.Join(", ", CanonicalGrammar.DtypeTokens.Values.Distinct())})");
        return "ndarray[" + Describe(elementType) + "]";
    }

    private string RenderLiteralValue(TypedConstant value)
    {
        if (value.Kind == TypedConstantKind.Array)
        {
            Reject("a Literal option cannot itself be an array");
            return "?";
        }

        if (value.IsNull)
        {
            return CanonicalGrammar.NoneToken;
        }

        // Enum-valued Literal members have no C# spelling in grammar version 1: the attribute
        // boxes them to their underlying integer, which would silently render as the distinct
        // schema Literal[0] (§8).
        if (value.Kind == TypedConstantKind.Enum || value.Kind == TypedConstantKind.Type)
        {
            Reject($"a Literal option of kind {value.Kind} cannot be expressed; options must be string, int, bool, or null");
            return "?";
        }

        switch (value.Value)
        {
            // bool before the integral cases: True/False is its own rendering and never 1/0.
            case bool flag:
                return flag ? "True" : "False";
            case string text:
                return CanonicalGrammar.QuoteLiteralString(text);

            // A char erases to a one-character string, exactly as the `char` scalar erases to
            // `str`. Literal['a'] is the same schema whichever way C# spells the option.
            case char character:
                return CanonicalGrammar.QuoteLiteralString(character.ToString());
            case sbyte or byte or short or ushort or int or uint or long or ulong:
                return System.Convert.ToString(value.Value, CultureInfo.InvariantCulture) ?? "?";
            default:
                Reject(
                    $"the Literal option '{value.Value}' of type "
                    + $"{(value.Type is null ? "?" : Describe(value.Type))} cannot be expressed; "
                    + "options must be string, int, bool, or null");
                return "?";
        }
    }

    private static string Describe(ITypeSymbol type) => type.ToDisplayString(_describeFormat);

    private void Reject(string problem) => _problems.Add(problem);
}
