using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Versionable.Analyzers.Grammar;

/// <summary>
/// Resolves a <c>[Versionable]</c> type symbol into a <see cref="SchemaModel"/>: its
/// serializable members, their canonical renderings, the payload and hash they produce, and
/// every construct the grammar rejects.
/// </summary>
/// <remarks>
/// Python counterpart: <c>_resolveFields()</c> plus the validation branch of
/// <c>Versionable.__init_subclass__</c> in <c>src/versionable/_base.py</c>. Python reads
/// dataclass fields; C# has no equivalent marker, so the rule is spelled out here — see
/// <see cref="CollectFields"/>.
/// </remarks>
internal static class SchemaModelBuilder
{
    /// <summary>Fully-qualified name of the attribute that marks a persistable root.</summary>
    internal const string VersionableAttributeName = "Versionable.VersionableAttribute";

    internal const string FieldAttributeName = "Versionable.VersionableFieldAttribute";
    internal const string LiteralValuesAttributeName = "Versionable.LiteralValuesAttribute";
    internal const string MigrationAttributeName = "Versionable.Migrations.MigrationAttribute";
    internal const string IgnoreAttributeName = "Versionable.VersionableIgnoreAttribute";
    internal const string MigrateClassName = "Migrate";
    internal const string MigrationTypeName = "Versionable.Migrations.Migration";
    internal const string MigrationContextTypeName = "Versionable.Migrations.MigrationContext";
    internal const string MigrationChainTypeName = "Versionable.Migrations.IMigrationChain";

    /// <summary>Builds the model for a <c>[Versionable]</c> type.</summary>
    /// <param name="type">The annotated type.</param>
    /// <param name="attribute">Its <c>[Versionable]</c> attribute data.</param>
    /// <returns>The resolved model, including any problems found.</returns>
    internal static SchemaModel Build(INamedTypeSymbol type, AttributeData attribute)
    {
        List<SchemaProblem> problems = new();
        Location typeLocation = type.Locations.FirstOrDefault() ?? Location.None;
        SchemaModel model = new()
        {
            Type = type,
            SerializationName = CanonicalTypeRenderer.SerializationName(type),
            IsPartial = IsPartialEverywhere(type),
            IsAbstract = type.IsAbstract,
            SupportsModuleInitializer = SupportsModuleInitializer(type),
            HidesBaseMembers = BaseCarriesGeneratedMembers(type),
        };

        ReadAttribute(attribute, model);

        if (!model.IsPartial)
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.MustBePartial,
                typeLocation,
                type.Name,
                IsPartialDeclaration(type) ? "an enclosing type" : "it"));
        }

        for (INamedTypeSymbol? enclosing = type; enclosing is not null; enclosing = enclosing.ContainingType)
        {
            if (!enclosing.IsGenericType)
            {
                continue;
            }

            problems.Add(new SchemaProblem(
                VersionableDiagnostics.UnsupportedType,
                typeLocation,
                type.Name,
                type.Name,
                ReferenceEquals(enclosing, type)
                    ? "a generic type has no Serialization Name that survives the grammar's parameter dropping (§9)"
                    : $"it is nested in the generic type '{enclosing.Name}', which has no single metadata instance"));
            break;
        }

        ImmutableArray<SchemaField> fields = CollectFields(type, problems);
        model.Fields = fields;
        model.MigrateTypeIsChain = MigrateTypeIsChain(type);

        // A Migrate class that is itself a chain owns its version list outright: the members this
        // would collect are that chain's implementation detail, and its FromVersions is a run-time
        // value no compile-time check can read.
        model.Migrations = model.MigrateTypeIsChain
            ? ImmutableArray<SchemaMigration>.Empty
            : CollectMigrations(type, problems);
        model.ReferencedTypes = CollectReferencedTypes(fields);

        ValidateWireNames(type, fields, problems);
        CheckNullableContext(type, fields, typeLocation, problems);
        CheckMigrationContiguity(type, model.Migrations, typeLocation, problems);

        model.Payload = SchemaHash.ComputePayload(
            fields.Select(field => new KeyValuePair<string, string>(field.WireName, field.CanonicalType)));
        model.ComputedHash = SchemaHash.ComputeHash(model.Payload);

        if (!string.Equals(model.DeclaredHash, model.ComputedHash, StringComparison.Ordinal))
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.HashMismatch,
                AttributeLocation(attribute) ?? typeLocation,
                type.Name,
                model.DeclaredHash,
                model.ComputedHash,
                model.Payload));
        }

        // An abstract type is a polymorphic base: a load resolves the concrete type from the
        // value's envelope and never constructs the base, so requiring a constructor for it
        // would reject exactly the declaration the grammar blesses for polymorphic fields.
        string factoryFailure = string.Empty;
        model.Factory = type.IsAbstract ? null : PlanFactory(type, fields, out factoryFailure);
        if (!type.IsAbstract && model.Factory is null)
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.NoUsableConstructor,
                typeLocation,
                type.Name,
                factoryFailure));
        }

        if (!model.SupportsModuleInitializer && model.IsPartial)
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.NotSelfRegistering,
                typeLocation,
                type.Name,
                ModuleInitializerObstacle(type)));
        }

        model.Problems = problems.ToImmutableArray();
        return model;
    }

    /// <summary>Finds the <c>[Versionable]</c> attribute on a type, if any.</summary>
    /// <param name="type">Any named type.</param>
    /// <returns>The attribute data, or <see langword="null"/>.</returns>
    internal static AttributeData? FindVersionableAttribute(INamedTypeSymbol type) =>
        FindAttribute(type, VersionableAttributeName);

    private static void ReadAttribute(AttributeData attribute, SchemaModel model)
    {
        foreach (KeyValuePair<string, TypedConstant> argument in attribute.NamedArguments)
        {
            switch (argument.Key)
            {
                case "Version":
                    model.Version = argument.Value.Value as int? ?? 0;
                    break;
                case "Hash":
                    model.DeclaredHash = argument.Value.Value as string ?? string.Empty;
                    break;
                case "OldNames":
                    model.OldNames = argument.Value.Values
                        .Select(value => value.Value as string)
                        .Where(value => value is not null)
                        .Select(value => value!)
                        .ToImmutableArray();
                    break;
                case "Register":
                    model.Register = argument.Value.Value as bool? ?? true;
                    break;
                case "SkipDefaults":
                    model.SkipDefaults = argument.Value.Value as bool? ?? false;
                    break;
                case "ValidateLiterals":
                    model.ValidateLiterals = argument.Value.Value as bool? ?? true;
                    break;
                case "Unknown":
                    model.UnknownPolicy = UnknownPolicyName(argument.Value);
                    break;
                default:
                    break;
            }
        }
    }

    private static string UnknownPolicyName(TypedConstant value) => (value.Value as int?) switch
    {
        1 => "Error",
        2 => "Preserve",
        _ => "Ignore",
    };

    /// <summary>
    /// Collects the serializable members of <paramref name="type"/>, base declarations first.
    /// </summary>
    /// <remarks>
    /// The C# analogue of "a dataclass field": a public, non-static property or field that
    /// holds state. Computed members are excluded — a get-only property with a body is a
    /// derived value, not state, and including it would both change the hash and break the
    /// generated factory. A get-only <em>auto</em> property is state and is included; it is
    /// how init-only and positional-record members are declared.
    /// </remarks>
    /// <param name="type">The annotated type.</param>
    /// <param name="problems">Collector for grammar violations.</param>
    /// <returns>The fields, in declaration order.</returns>
    private static ImmutableArray<SchemaField> CollectFields(INamedTypeSymbol type, List<SchemaProblem> problems)
    {
        List<SchemaField> fields = new();
        Dictionary<string, int> byName = new(StringComparer.Ordinal);

        foreach (INamedTypeSymbol declaring in BaseFirst(type))
        {
            foreach (ISymbol member in declaring.GetMembers())
            {
                SchemaField? field = TryDescribe(member, type, problems);
                if (field is null)
                {
                    continue;
                }

                // An override or a `new` member replaces the base declaration in place, so
                // declaration order stays base-first while the accessors come from the most
                // derived declaration.
                if (byName.TryGetValue(field.ClrName, out int existing))
                {
                    fields[existing] = field;
                    continue;
                }

                byName[field.ClrName] = fields.Count;
                fields.Add(field);
            }
        }

        return fields.ToImmutableArray();
    }

    private static IEnumerable<INamedTypeSymbol> BaseFirst(INamedTypeSymbol type)
    {
        Stack<INamedTypeSymbol> chain = new();
        for (INamedTypeSymbol? current = type;
             current is not null && current.SpecialType != SpecialType.System_Object;
             current = current.BaseType)
        {
            chain.Push(current);
        }

        return chain;
    }

    private static SchemaField? TryDescribe(ISymbol member, INamedTypeSymbol owner, List<SchemaProblem> problems)
    {
        // The opt-out, checked first: an ignored member leaves the payload and the emitted
        // descriptors together, so nothing downstream can see it at all.
        if (FindAttribute(member, IgnoreAttributeName) is not null)
        {
            return null;
        }

        ITypeSymbol memberType;
        bool initializable;
        bool assignable;

        switch (member)
        {
            case IPropertySymbol property:
                if (property.IsStatic
                    || property.IsIndexer
                    || property.GetMethod is null
                    || property.DeclaredAccessibility != Accessibility.Public
                    || property.Name == "EqualityContract"
                    || IsComputed(property)
                    || !IsReachableFrom(property.GetMethod, owner))
                {
                    return null;
                }

                memberType = property.Type;
                initializable = property.SetMethod is not null && IsReachableFrom(property.SetMethod, owner);
                assignable = initializable
                    && property.SetMethod is { IsInitOnly: false }
                    && !owner.IsValueType;
                break;

            case IFieldSymbol candidate:
                if (candidate.IsStatic
                    || candidate.IsConst
                    || candidate.IsImplicitlyDeclared
                    || candidate.DeclaredAccessibility != Accessibility.Public)
                {
                    return null;
                }

                memberType = candidate.Type;
                initializable = !candidate.IsReadOnly;
                assignable = !candidate.IsReadOnly && !owner.IsValueType;
                break;

            default:
                return null;
        }

        Location location = member.Locations.FirstOrDefault() ?? Location.None;
        CanonicalTypeRenderer renderer = new();
        AttributeData? literalValues = FindAttribute(member, LiteralValuesAttributeName);
        ImmutableArray<TypedConstant> options = literalValues is null
            ? ImmutableArray<TypedConstant>.Empty
            : LiteralOptions(literalValues);

        string canonical = literalValues is null
            ? renderer.Render(memberType)
            : renderer.RenderLiteral(memberType, options);

        foreach (string problem in renderer.Problems)
        {
            problems.Add(new SchemaProblem(
                literalValues is null
                    ? VersionableDiagnostics.UnsupportedType
                    : VersionableDiagnostics.UnsupportedLiteralValue,
                location,
                member.Name,
                owner.Name,
                problem));
        }

        AttributeData? fieldAttribute = FindAttribute(member, FieldAttributeName);
        string wireName = fieldAttribute is not null
            && fieldAttribute.ConstructorArguments.Length == 1
            && fieldAttribute.ConstructorArguments[0].Value is string declared
                ? declared
                : member.Name;

        bool hasFallback = literalValues is not null
            && literalValues.NamedArguments.Any(argument => argument.Key == "Fallback");

        return new SchemaField
        {
            Member = member,
            WireName = wireName,
            ClrName = member.Name,
            ClrType = memberType,
            CanonicalType = canonical,
            IsInitializable = initializable,
            IsAssignable = assignable,
            IsLiteral = literalValues is not null,
            LiteralValues = options,
            HasLiteralFallback = hasFallback,
            LiteralFallback = hasFallback
                ? literalValues!.NamedArguments.First(argument => argument.Key == "Fallback").Value
                : default,
            DefaultExpression = ConstantDefaultExpression(member, memberType),
            Location = location,
        };
    }

    private static ImmutableArray<TypedConstant> LiteralOptions(AttributeData attribute)
    {
        if (attribute.ConstructorArguments.Length != 1)
        {
            return ImmutableArray<TypedConstant>.Empty;
        }

        TypedConstant argument = attribute.ConstructorArguments[0];
        return argument.Kind == TypedConstantKind.Array && !argument.IsNull
            ? argument.Values
            : ImmutableArray<TypedConstant>.Empty;
    }

    /// <summary>
    /// Whether generated code sitting inside <paramref name="owner"/> can call
    /// <paramref name="accessor"/>.
    /// </summary>
    /// <remarks>
    /// A <c>public int X { get; private set; }</c> — or the rarer <c>{ private get; set; }</c> —
    /// declared on a <em>base</em> type is invisible from the derived type's own partial, so
    /// emitting an accessor for it would produce generated code that does not compile. On the
    /// owner itself every accessor is reachable, private ones included.
    /// </remarks>
    private static bool IsReachableFrom(IMethodSymbol accessor, INamedTypeSymbol owner)
    {
        if (SymbolEqualityComparer.Default.Equals(accessor.ContainingType, owner))
        {
            return true;
        }

        return accessor.DeclaredAccessibility switch
        {
            Accessibility.Public or Accessibility.Protected or Accessibility.ProtectedOrInternal => true,
            Accessibility.Internal or Accessibility.ProtectedAndInternal =>
                SymbolEqualityComparer.Default.Equals(accessor.ContainingAssembly, owner.ContainingAssembly),
            _ => false,
        };
    }

    /// <summary>
    /// Whether the property computes its value rather than storing it: an expression-bodied
    /// property, or a get accessor with a body, and no setter to write through.
    /// </summary>
    private static bool IsComputed(IPropertySymbol property)
    {
        if (property.SetMethod is not null)
        {
            return false;
        }

        foreach (SyntaxReference reference in property.DeclaringSyntaxReferences)
        {
            if (reference.GetSyntax() is not PropertyDeclarationSyntax declaration)
            {
                continue;
            }

            if (declaration.ExpressionBody is not null)
            {
                return true;
            }

            if (declaration.AccessorList is null)
            {
                continue;
            }

            foreach (AccessorDeclarationSyntax accessor in declaration.AccessorList.Accessors)
            {
                if (accessor.Body is not null || accessor.ExpressionBody is not null)
                {
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>
    /// The C# source of a closure-free initializer, or <see langword="null"/>.
    /// </summary>
    /// <remarks>
    /// The generated <c>DefaultFactory</c> is a <c>static</c> lambda, so the one thing an
    /// initializer must not do is close over state — an instance member or a
    /// primary-constructor parameter is simply not in scope there. Everything that closes over
    /// nothing is re-emittable: literals, <c>default</c>, and empty construction (<c>new()</c>,
    /// <c>[]</c>, <c>new List&lt;T&gt;()</c>), which is re-emitted as an explicit
    /// <c>new</c> of the member's own type so the generated file needs no <c>using</c>
    /// directives. An empty collection literal is rebuilt per call rather than shared, which is
    /// the whole reason the contract makes this a delegate and not a constant.
    /// <para>
    /// Anything else reports <c>HasDefault = false</c>. That never affects a hash; it affects
    /// <c>SkipDefaults</c> and the TOML/YAML <c>CommentDefaults</c> option.
    /// </para>
    /// </remarks>
    private static string? ConstantDefaultExpression(ISymbol member, ITypeSymbol memberType)
    {
        string declared = memberType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);

        // `default(T)` names the annotation for a reference type, because the generated file is
        // always `#nullable enable`: `default(Foo)` there is a maybe-null value of a
        // non-nullable type, which is CS8600 the moment it is converted. The display format
        // drops the annotation deliberately (it has to keep `typeof(...)` legal), so it is added
        // back here rather than taken from it. Value types keep their exact spelling — widening
        // `int` to `int?` would make `= default` mean null instead of 0.
        string declaredDefault = memberType.IsReferenceType
            && memberType.NullableAnnotation != NullableAnnotation.NotAnnotated
                ? declared + "?"
                : declared;

        foreach (SyntaxReference reference in member.DeclaringSyntaxReferences)
        {
            ExpressionSyntax? initializer = reference.GetSyntax() switch
            {
                PropertyDeclarationSyntax property => property.Initializer?.Value,
                VariableDeclaratorSyntax variable => variable.Initializer?.Value,
                ParameterSyntax parameter => parameter.Default?.Value,
                _ => null,
            };

            switch (initializer)
            {
                case null:
                    continue;
                case LiteralExpressionSyntax literal when literal.IsKind(SyntaxKind.DefaultLiteralExpression):
                    return "default(" + declaredDefault + ")";
                case LiteralExpressionSyntax literal:
                    return literal.Token.Text;
                case DefaultExpressionSyntax:
                    return "default(" + declaredDefault + ")";
                case PrefixUnaryExpressionSyntax negation
                    when negation.IsKind(SyntaxKind.UnaryMinusExpression)
                        && negation.Operand is LiteralExpressionSyntax number:
                    return "-" + number.Token.Text;
                case ImplicitObjectCreationExpressionSyntax { Initializer: null } implicitNew
                    when implicitNew.ArgumentList.Arguments.Count == 0:
                case ObjectCreationExpressionSyntax { Initializer: null } explicitNew
                    when explicitNew.ArgumentList is null || explicitNew.ArgumentList.Arguments.Count == 0:
                case CollectionExpressionSyntax { Elements.Count: 0 }:
                    return EmptyConstruction(memberType, declared);
                default:
                    continue;
            }
        }

        return null;
    }

    /// <summary>
    /// Re-emits an empty-construction initializer as an explicit expression of the member's
    /// own type, or <see langword="null"/> when the type cannot be constructed that way.
    /// </summary>
    private static string? EmptyConstruction(ITypeSymbol memberType, string declared)
    {
        if (memberType is IArrayTypeSymbol array)
        {
            return "global::System.Array.Empty<"
                + array.ElementType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) + ">()";
        }

        // FrozenSet<T> and friends are abstract or factory-built: `= []` compiles through a
        // collection builder the generator cannot name, so those keep HasDefault = false.
        if (memberType is not INamedTypeSymbol named || named.IsAbstract || named.IsStatic)
        {
            return null;
        }

        foreach (IMethodSymbol constructor in named.InstanceConstructors)
        {
            if (constructor.Parameters.Length == 0
                && constructor.DeclaredAccessibility is Accessibility.Public or Accessibility.Internal)
            {
                return "new " + declared + "()";
            }
        }

        return null;
    }

    private static void ValidateWireNames(
        INamedTypeSymbol type,
        ImmutableArray<SchemaField> fields,
        List<SchemaProblem> problems)
    {
        HashSet<string> seen = new(StringComparer.Ordinal);
        foreach (SchemaField field in fields)
        {
            int reserved = field.WireName.IndexOfAny(CanonicalGrammar.ReservedWireNameCharacters);
            if (reserved >= 0)
            {
                problems.Add(new SchemaProblem(
                    VersionableDiagnostics.InvalidWireName,
                    field.Location,
                    field.WireName,
                    type.Name,
                    $"'{field.WireName[reserved]}' is reserved by the hash payload"));
            }
            else if (field.WireName.Length == 0)
            {
                problems.Add(new SchemaProblem(
                    VersionableDiagnostics.InvalidWireName,
                    field.Location,
                    field.WireName,
                    type.Name,
                    "a wire name cannot be empty"));
            }

            foreach (char character in field.WireName)
            {
                if (char.IsSurrogate(character))
                {
                    problems.Add(new SchemaProblem(
                        VersionableDiagnostics.InvalidWireName,
                        field.Location,
                        field.WireName,
                        type.Name,
                        "grammar version 1 restricts wire names to the Basic Multilingual Plane, so that "
                            + "ordinal and code-point ordering agree"));
                    break;
                }
            }

            if (!seen.Add(field.WireName))
            {
                problems.Add(new SchemaProblem(
                    VersionableDiagnostics.DuplicateWireName,
                    field.Location,
                    field.WireName,
                    type.Name));
            }
        }
    }

    private static void CheckNullableContext(
        INamedTypeSymbol type,
        ImmutableArray<SchemaField> fields,
        Location location,
        List<SchemaProblem> problems)
    {
        foreach (SchemaField field in fields)
        {
            if (field.ClrType.IsReferenceType && field.ClrType.NullableAnnotation == NullableAnnotation.None)
            {
                problems.Add(new SchemaProblem(
                    VersionableDiagnostics.NullableContextDisabled,
                    location,
                    type.Name));
                return;
            }
        }
    }

    /// <summary>
    /// Collects the migrations declared as members of the nested <c>Migrate</c> class, reporting
    /// members that name themselves migrations but cannot be run as one.
    /// </summary>
    /// <remarks>
    /// Python counterpart: the <c>dir(migrateClass)</c> walk in <c>resolveMigrations</c>
    /// (<c>src/versionable/_migration.py</c>), which matches <c>vN</c> attributes holding a
    /// <c>Migration</c> and <c>@migration</c>-decorated functions. Python resolves at load and
    /// simply ignores a member of the wrong kind; here the generator has to emit a reference to
    /// each member, so a member that cannot be referenced is reported rather than dropped —
    /// silently dropping it would turn a typo into a load-time failure on old files only.
    /// </remarks>
    private static ImmutableArray<SchemaMigration> CollectMigrations(
        INamedTypeSymbol type,
        List<SchemaProblem> problems)
    {
        INamedTypeSymbol? migrate = type.GetTypeMembers(MigrateClassName).FirstOrDefault();
        if (migrate is null)
        {
            return ImmutableArray<SchemaMigration>.Empty;
        }

        List<SchemaMigration> migrations = new();
        foreach (ISymbol member in migrate.GetMembers())
        {
            if (member is IMethodSymbol method)
            {
                CollectImperativeMigration(type, method, migrations, problems);
                continue;
            }

            // Declarative migrations are fields or properties named V1, V2, … — the source
            // version is in the name, exactly as Python's Migrate.v1 carries it.
            if (member is IFieldSymbol or IPropertySymbol && TryParseVersionMember(member.Name, out int version))
            {
                CollectDeclarativeMigration(type, member, version, migrations, problems);
            }
        }

        migrations.Sort((left, right) => left.FromVersion.CompareTo(right.FromVersion));
        return migrations.ToImmutableArray();
    }

    private static void CollectDeclarativeMigration(
        INamedTypeSymbol type,
        ISymbol member,
        int version,
        List<SchemaMigration> migrations,
        List<SchemaProblem> problems)
    {
        ITypeSymbol memberType = member is IFieldSymbol field ? field.Type : ((IPropertySymbol)member).Type;
        Location location = member.Locations.FirstOrDefault() ?? Location.None;

        if (CanonicalTypeRenderer.FullName(memberType) != MigrationTypeName)
        {
            // Not a diagnostic about naming: a V-named member of the Migrate class is claiming to
            // be the migration for that version, and one that is not a Migration leaves the chain
            // with a hole nothing else reports.
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.MigrationChainInvalid,
                location,
                type.Name,
                $"'{MigrateClassName}.{member.Name}' is of type '{memberType.ToDisplayString()}'; a "
                    + $"declarative migration must be a '{MigrationTypeName}'"));
            return;
        }

        if (!member.IsStatic)
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.MigrationChainInvalid,
                location,
                type.Name,
                $"'{MigrateClassName}.{member.Name}' must be static; the chain is read without "
                    + "constructing anything"));
            return;
        }

        if (!IsReachableFromDeclaringType(member))
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.MigrationChainInvalid,
                location,
                type.Name,
                $"'{MigrateClassName}.{member.Name}' is private to '{MigrateClassName}', so the "
                    + $"metadata generated for '{type.Name}' cannot read it"));
            return;
        }

        if (member is IPropertySymbol property && property.GetMethod is null)
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.MigrationChainInvalid,
                location,
                type.Name,
                $"'{MigrateClassName}.{member.Name}' has no getter"));
            return;
        }

        migrations.Add(new SchemaMigration(version, member.Name, imperative: false));
    }

    private static void CollectImperativeMigration(
        INamedTypeSymbol type,
        IMethodSymbol method,
        List<SchemaMigration> migrations,
        List<SchemaProblem> problems)
    {
        AttributeData? attribute = FindAttribute(method, MigrationAttributeName);
        if (attribute is null)
        {
            return;
        }

        Location location = method.Locations.FirstOrDefault() ?? Location.None;
        int? fromVersion = null;
        foreach (KeyValuePair<string, TypedConstant> argument in attribute.NamedArguments)
        {
            if (argument.Key == "FromVersion" && argument.Value.Value is int declared)
            {
                fromVersion = declared;
            }
        }

        if (fromVersion is null)
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.MigrationChainInvalid,
                location,
                type.Name,
                $"'{MigrateClassName}.{method.Name}' carries [Migration] without a FromVersion, so "
                    + "there is no version it migrates from"));
            return;
        }

        string signature =
            $"static void {method.Name}({MigrationContextTypeName} data)";
        bool shaped = method.IsStatic
            && !method.IsGenericMethod
            && method.ReturnsVoid
            && method.Parameters.Length == 1
            && method.Parameters[0].RefKind == RefKind.None
            && CanonicalTypeRenderer.FullName(method.Parameters[0].Type) == MigrationContextTypeName;

        if (!shaped)
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.MigrationChainInvalid,
                location,
                type.Name,
                $"'{MigrateClassName}.{method.Name}' carries [Migration] but is not shaped like one; "
                    + $"it must be declared '{signature}'"));
            return;
        }

        if (!IsReachableFromDeclaringType(method))
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.MigrationChainInvalid,
                location,
                type.Name,
                $"'{MigrateClassName}.{method.Name}' is private to '{MigrateClassName}', so the "
                    + $"metadata generated for '{type.Name}' cannot call it"));
            return;
        }

        migrations.Add(new SchemaMigration(fromVersion.Value, method.Name, imperative: true));
    }

    /// <summary>
    /// Whether a member of the nested <c>Migrate</c> class can be referenced from the generated
    /// part of the declaring type.
    /// </summary>
    /// <remarks>
    /// The generated code is another part of the <c>[Versionable]</c> type, so it sees everything
    /// that type sees — which is everything except members private to <c>Migrate</c> itself:
    /// private members of a nested type are accessible within that type, not from the type it is
    /// nested in.
    /// </remarks>
    private static bool IsReachableFromDeclaringType(ISymbol member) =>
        member.DeclaredAccessibility != Accessibility.Private;

    /// <summary>
    /// Whether some base type already has the generated <c>VersionableMetadata</c> member, either
    /// because it is <c>[Versionable]</c> in this compilation or because it implements
    /// <c>IVersionableMetadataProvider</c> from a referenced assembly.
    /// </summary>
    private static bool BaseCarriesGeneratedMembers(INamedTypeSymbol type)
    {
        for (INamedTypeSymbol? current = type.BaseType; current is not null; current = current.BaseType)
        {
            if (current.SpecialType == SpecialType.System_Object)
            {
                return false;
            }

            // Only when the base is certain to have the member, for the same reason
            // VersionableMetadataGenerator.TryVersionableMetadata is careful: a base that is
            // [Versionable] but not partial generates nothing, and `new` against an absent member
            // is CS0109 — which would bury the base's own VSN0005 under a compiler error.
            bool carries = current.DeclaringSyntaxReferences.Length == 0
                ? current.AllInterfaces.Any(contract =>
                    CanonicalTypeRenderer.FullName(contract) == "Versionable.IVersionableMetadataProvider")
                : FindVersionableAttribute(current) is not null && IsPartialEverywhere(current);

            if (carries)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Whether the nested <c>Migrate</c> type is itself an instantiable
    /// <c>IMigrationChain</c> the generator can hand to
    /// <c>VersionableMetadata.Migrations</c>.
    /// </summary>
    /// <remarks>
    /// The two declaration forms are checked in order rather than assumed disjoint. A
    /// <c>static class Migrate</c> holding <c>V1</c>/<c>V2</c> builder members can neither
    /// implement an interface nor be constructed, so that shape can only ever be the composed
    /// one — but a <c>Migrate</c> that <em>does</em> implement <c>IMigrationChain</c> may also
    /// hold builder members, and then the chain type wins: it is the more explicit statement, and
    /// its members are its own implementation detail. <c>Build</c> acts on that by skipping member
    /// collection and the contiguity check entirely when this returns true, which is also why a
    /// run-time chain is the one form the analyzer leaves unverified — its
    /// <c>FromVersions</c> is a run-time value.
    /// </remarks>
    private static bool MigrateTypeIsChain(INamedTypeSymbol type)
    {
        INamedTypeSymbol? migrate = type.GetTypeMembers(MigrateClassName).FirstOrDefault();
        if (migrate is null || migrate.IsAbstract || migrate.IsStatic || migrate.IsGenericType)
        {
            return false;
        }

        if (!migrate.AllInterfaces.Any(contract =>
            CanonicalTypeRenderer.FullName(contract) == MigrationChainTypeName))
        {
            return false;
        }

        // A parameterless constructor is required — the generator has no arguments to supply — and
        // a private one is deliberately not accepted. Generated code does sit inside the declaring
        // type, so it could call a private constructor, but a type that hides its constructor is
        // saying it is not to be instantiated from outside its own members; taking the
        // conservative reading here means the type simply gets no chain rather than getting one it
        // did not offer.
        return migrate.InstanceConstructors.Any(constructor =>
            constructor.Parameters.Length == 0
            && constructor.DeclaredAccessibility != Accessibility.Private);
    }

    private static bool TryParseVersionMember(string name, out int version)
    {
        version = 0;
        if (name.Length < 2 || name[0] != 'V')
        {
            return false;
        }

        for (int i = 1; i < name.Length; i++)
        {
            if (!char.IsDigit(name[i]))
            {
                return false;
            }
        }

        return int.TryParse(name.Substring(1), NumberStyles.None, CultureInfo.InvariantCulture, out version);
    }

    private static void CheckMigrationContiguity(
        INamedTypeSymbol type,
        ImmutableArray<SchemaMigration> migrations,
        Location location,
        List<SchemaProblem> problems)
    {
        if (migrations.Length == 0)
        {
            return;
        }

        // Both declaration forms are in here: the check is over the chain, not over one syntax,
        // because a chain half declarative and half imperative is a chain like any other.
        ImmutableArray<int> versions = migrations.Select(migration => migration.FromVersion).ToImmutableArray();

        List<int> duplicates = versions.GroupBy(version => version)
            .Where(group => group.Count() > 1)
            .Select(group => group.Key)
            .ToList();
        if (duplicates.Count > 0)
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.MigrationChainInvalid,
                location,
                type.Name,
                "two migrations declare FromVersion "
                    + string.Join(", ", duplicates.Select(version => version.ToString(CultureInfo.InvariantCulture)))));
        }

        // Bounded by the declared migrations at both ends, deliberately. The gap between the
        // newest declared migration and Version - 1 is left unchecked because a version bumped
        // twice before release — or bumped for a change no data needs migrating for — is
        // legitimate, and the load path refuses the files it actually cannot read anyway. Only
        // the gaps between two declared migrations are unambiguously a mistake.
        List<int> missing = new();
        for (int version = versions[0]; version < versions[versions.Length - 1]; version++)
        {
            if (!versions.Contains(version))
            {
                missing.Add(version);
            }
        }

        if (missing.Count > 0)
        {
            problems.Add(new SchemaProblem(
                VersionableDiagnostics.MigrationChainInvalid,
                location,
                type.Name,
                "no migration declared from version "
                    + string.Join(", ", missing.Select(version => version.ToString(CultureInfo.InvariantCulture)))
                    + ", but a later one is (gaps below the oldest migration are fine; gaps above it are not)"));
        }
    }

    /// <summary>
    /// Collects the named types reachable from <paramref name="fields"/> that claim a
    /// Serialization Name, for the uniqueness check of GRAMMAR §9.
    /// </summary>
    private static ImmutableArray<INamedTypeSymbol> CollectReferencedTypes(ImmutableArray<SchemaField> fields)
    {
        List<INamedTypeSymbol> collected = new();
        foreach (SchemaField field in fields)
        {
            Collect(field.ClrType, collected);
        }

        return collected.ToImmutableArray();

        static void Collect(ITypeSymbol type, List<INamedTypeSymbol> into)
        {
            if (CanonicalTypeRenderer.TryUnwrapNullable(type, out ITypeSymbol inner))
            {
                Collect(inner, into);
                return;
            }

            if (type is IArrayTypeSymbol array)
            {
                if (array.ElementType.SpecialType != SpecialType.System_Byte)
                {
                    Collect(array.ElementType, into);
                }

                return;
            }

            if (type is not INamedTypeSymbol named)
            {
                return;
            }

            if (named.IsTupleType)
            {
                foreach (IFieldSymbol element in named.TupleElements)
                {
                    Collect(element.Type, into);
                }

                return;
            }

            if (named.SpecialType != SpecialType.None)
            {
                return;
            }

            string full = CanonicalTypeRenderer.FullName(named);
            if (CanonicalGrammar.ConverterNames.ContainsKey(full)
                || full == "System.Half"
                || full == "System.Numerics.Complex"
                || full == "System.Numerics.Tensors.Tensor")
            {
                return;
            }

            switch (full)
            {
                case "System.ValueTuple":
                case "System.Collections.Generic.List":
                case "System.Collections.Generic.HashSet":
                case "System.Collections.Generic.Dictionary":
                case "System.Collections.Frozen.FrozenSet":
                case "System.Collections.Immutable.ImmutableHashSet":
                    foreach (ITypeSymbol argument in named.TypeArguments)
                    {
                        Collect(argument, into);
                    }

                    return;
                default:
                    into.Add(named.OriginalDefinition);
                    return;
            }
        }
    }

    private static FactoryPlan? PlanFactory(
        INamedTypeSymbol type,
        ImmutableArray<SchemaField> fields,
        out string failure)
    {
        FactoryPlan? best = null;
        List<string> rejections = new();

        foreach (IMethodSymbol constructor in type.InstanceConstructors)
        {
            if (constructor.IsStatic || constructor.DeclaredAccessibility == Accessibility.NotApplicable)
            {
                continue;
            }

            int[] parameterFields = new int[constructor.Parameters.Length];
            bool[] covered = new bool[fields.Length];
            bool usable = true;

            for (int i = 0; i < constructor.Parameters.Length; i++)
            {
                string parameterName = constructor.Parameters[i].Name;
                int match = IndexOfField(fields, parameterName, out string? ambiguity);
                if (ambiguity is not null)
                {
                    rejections.Add(
                        $"'{constructor.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat)}' takes "
                        + $"'{parameterName}', which matches {ambiguity} once case is folded — rename the "
                        + "parameter to match one of them exactly");
                    usable = false;
                    break;
                }

                if (match < 0)
                {
                    rejections.Add(
                        $"'{constructor.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat)}' takes "
                        + $"'{parameterName}', which is not a serializable member");
                    usable = false;
                    break;
                }

                parameterFields[i] = match;
                covered[match] = true;
            }

            if (!usable)
            {
                continue;
            }

            List<int> initializerFields = new();
            for (int i = 0; i < fields.Length; i++)
            {
                if (covered[i])
                {
                    continue;
                }

                if (!fields[i].IsInitializable)
                {
                    rejections.Add(
                        $"'{constructor.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat)}' leaves "
                        + $"'{fields[i].ClrName}' unset, and it has no settable accessor");
                    usable = false;
                    break;
                }

                initializerFields.Add(i);
            }

            if (!usable)
            {
                continue;
            }

            if (best is null || constructor.Parameters.Length > best.Constructor.Parameters.Length)
            {
                best = new FactoryPlan
                {
                    Constructor = constructor,
                    ParameterFields = parameterFields.ToImmutableArray(),
                    InitializerFields = initializerFields.ToImmutableArray(),
                };
            }
        }

        failure = best is null
            ? (rejections.Count == 0 ? "it declares no accessible constructor" : string.Join("; ", rejections))
            : string.Empty;
        return best;
    }

    /// <summary>
    /// Finds the member a constructor parameter feeds.
    /// </summary>
    /// <remarks>
    /// Exact case wins outright, so the ordinary <c>Name</c>/<c>name</c> pairing needs no
    /// tie-break. Case-insensitive matching is the fallback, and it is only allowed to resolve
    /// when exactly one member folds onto the parameter: C# is case-sensitive, so a type can
    /// legally declare both <c>Value</c> and <c>value</c>, and picking whichever came first
    /// would silently wire a constructor argument to the wrong field. That reports
    /// <see cref="VersionableDiagnostics.NoUsableConstructor"/> instead.
    /// </remarks>
    /// <param name="fields">The serializable members, in declaration order.</param>
    /// <param name="parameterName">The constructor parameter's name.</param>
    /// <param name="ambiguity">A description of the competing members, when the fold is ambiguous.</param>
    /// <returns>The member index, or -1 when nothing matches.</returns>
    private static int IndexOfField(
        ImmutableArray<SchemaField> fields,
        string parameterName,
        out string? ambiguity)
    {
        ambiguity = null;

        for (int i = 0; i < fields.Length; i++)
        {
            if (string.Equals(fields[i].ClrName, parameterName, StringComparison.Ordinal))
            {
                return i;
            }
        }

        int folded = -1;
        List<string> candidates = new();
        for (int i = 0; i < fields.Length; i++)
        {
            if (string.Equals(fields[i].ClrName, parameterName, StringComparison.OrdinalIgnoreCase))
            {
                folded = i;
                candidates.Add($"'{fields[i].ClrName}'");
            }
        }

        if (candidates.Count > 1)
        {
            ambiguity = string.Join(" and ", candidates);
            return -1;
        }

        return folded;
    }

    /// <summary>Whether the type and every enclosing type is declared <c>partial</c>.</summary>
    /// <param name="type">Any named type.</param>
    /// <returns><see langword="true"/> when a generated partial can be emitted for it.</returns>
    internal static bool IsPartialEverywhere(INamedTypeSymbol type)
    {
        for (INamedTypeSymbol? current = type; current is not null; current = current.ContainingType)
        {
            if (!IsPartialDeclaration(current))
            {
                return false;
            }
        }

        return true;
    }

    private static bool IsPartialDeclaration(INamedTypeSymbol type)
    {
        foreach (SyntaxReference reference in type.DeclaringSyntaxReferences)
        {
            if (reference.GetSyntax() is TypeDeclarationSyntax declaration
                && declaration.Modifiers.Any(SyntaxKind.PartialKeyword))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Whether a <c>[ModuleInitializer]</c> can be emitted: the method must be reachable from
    /// module scope and must not sit inside a generic type.
    /// </summary>
    private static string ModuleInitializerObstacle(INamedTypeSymbol type)
    {
        for (INamedTypeSymbol? current = type; current is not null; current = current.ContainingType)
        {
            if (current.IsGenericType)
            {
                return ReferenceEquals(current, type)
                    ? "it is generic"
                    : $"it is nested in the generic type '{current.Name}'";
            }

            if (current.DeclaredAccessibility is not (Accessibility.Public or Accessibility.Internal))
            {
                return ReferenceEquals(current, type)
                    ? $"it is declared {Describe(current.DeclaredAccessibility)}, which is not reachable from module scope"
                    : $"it is nested in '{current.Name}', which is declared {Describe(current.DeclaredAccessibility)} "
                        + "and so is not reachable from module scope";
            }
        }

        return "of its declaration";
    }

    private static string Describe(Accessibility accessibility) => accessibility switch
    {
        Accessibility.Private => "private",
        Accessibility.Protected => "protected",
        Accessibility.ProtectedOrInternal => "protected internal",
        Accessibility.ProtectedAndInternal => "private protected",
        _ => accessibility.ToString().ToLowerInvariant(),
    };

    private static bool SupportsModuleInitializer(INamedTypeSymbol type)
    {
        for (INamedTypeSymbol? current = type; current is not null; current = current.ContainingType)
        {
            if (current.IsGenericType)
            {
                return false;
            }

            if (current.DeclaredAccessibility is not (Accessibility.Public or Accessibility.Internal))
            {
                return false;
            }
        }

        return true;
    }

    private static AttributeData? FindAttribute(ISymbol symbol, string fullName)
    {
        foreach (AttributeData attribute in symbol.GetAttributes())
        {
            if (attribute.AttributeClass is not null
                && CanonicalTypeRenderer.FullName(attribute.AttributeClass) == fullName)
            {
                return attribute;
            }
        }

        return null;
    }

    private static Location? AttributeLocation(AttributeData attribute) =>
        attribute.ApplicationSyntaxReference is { } reference
            ? Location.Create(reference.SyntaxTree, reference.Span)
            : null;
}
