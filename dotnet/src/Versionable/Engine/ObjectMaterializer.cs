using System.Globalization;
using Versionable.Errors;

namespace Versionable.Engine;

/// <summary>
/// Turns a dictionary of raw field values into an instance: unknown-field policy, literal
/// validation, defaults for absent fields, then the generated factory.
/// </summary>
/// <remarks>
/// Python counterpart: the tail of <c>load()</c> in <c>src/versionable/_api.py</c> and of
/// <c>_deserializeVersionable</c> in <c>src/versionable/_types.py</c> — two copies of the same
/// sequence in Python, one here, because the only difference between a root object and a nested
/// one is where the version comes from.
/// </remarks>
internal static class ObjectMaterializer
{
    /// <summary>
    /// Reads a nested object: its own envelope, its own concrete type, its own migrations.
    /// </summary>
    /// <param name="wire">The nested wire dictionary, envelope included.</param>
    /// <param name="declared">Metadata of the declared field type; the polymorphic upper bound.</param>
    /// <param name="scope">Ambient read state.</param>
    /// <returns>The materialized instance.</returns>
    internal static object ReadNested(object? wire, VersionableMetadata declared, ReadScope scope)
    {
        IReadOnlyDictionary<string, object?> map = WireValues.AsMap(wire);
        EnvelopeMetadata envelope = EnvelopeCodec.Read(map);
        VersionableMetadata actual = ResolveConcrete(envelope, declared);

        int version;
        if (envelope.Version is null)
        {
            // A hand-written file, or one written before envelopes. Assuming current is the only
            // option that loads at all, but it silently skips migrations, so say so.
            VersionableLog.Warn(
                $"Nested {actual.Name}: no version in the envelope; assuming the current version "
                    + $"({actual.Version}). If this data was written by older code, the load may fail.");
            version = actual.Version;
        }
        else
        {
            version = envelope.Version.Value;
        }

        IDictionary<string, object?> fields = MigrationRunner.Run(
            EnvelopeCodec.Strip(map), actual, version, scope.UpgradeInPlace, $"nested {actual.Name}");

        // The enclosing Materialize narrowed this to the skips belonging to *this* object before
        // it read the field that led here. Passing null instead was the bug: a nested skip became
        // an absent field one level down, with the same misleading message.
        return Materialize(fields, actual, scope.LazyFields, scope);
    }

    /// <summary>Builds an instance from already-migrated field values.</summary>
    /// <param name="fields">Raw field values keyed by wire name, envelope keys already stripped.</param>
    /// <param name="metadata">Metadata of the type to build.</param>
    /// <param name="lazyFields">
    /// Wire names the backend deliberately left unmaterialized. A value present in
    /// <paramref name="fields"/> under such a name is the backend's own sentinel and passes
    /// straight through; a name with no value at all is handled by
    /// <see cref="SkippedFieldValue"/>, which is the difference between "you asked me not to read
    /// this" and "the file does not have this".
    /// </param>
    /// <param name="scope">Ambient read state.</param>
    /// <returns>The materialized instance.</returns>
    internal static object Materialize(
        IDictionary<string, object?> fields,
        VersionableMetadata metadata,
        IReadOnlySet<string>? lazyFields,
        ReadScope scope)
    {
        ApplyUnknownFieldPolicy(fields, metadata);

        object?[] values = new object?[metadata.Fields.Count];
        IReadOnlySet<string>? enclosingLazy = scope.LazyFields;
        string enclosingPath = scope.Path;
        try
        {
            for (int index = 0; index < metadata.Fields.Count; index++)
            {
                FieldDescriptor field = metadata.Fields[index];
                scope.Path = enclosingPath.Length == 0 ? field.WireName : $"{enclosingPath}/{field.WireName}";

                if (fields.TryGetValue(field.WireName, out object? raw))
                {
                    // Narrowed before reading, because reading may descend into a nested object
                    // that has to see its own skips and nothing else.
                    scope.LazyFields = ScopeInto(lazyFields, field);
                    values[index] = ReadField(field, raw, metadata, lazyFields, scope);
                }
                else if (lazyFields is not null && lazyFields.Contains(field.WireName))
                {
                    // Absent from Fields but named in LazyFields: the backend was told not to read
                    // it. That is not the same thing as the file not having it, and reporting it as
                    // "missing from the file and has no default" would send a reader looking for a
                    // corrupt file instead of at the option they passed.
                    values[index] = SkippedFieldValue(field, metadata, scope.Path);
                }
                else
                {
                    values[index] = DefaultFor(field, metadata);
                }
            }
        }
        finally
        {
            scope.LazyFields = enclosingLazy;
            scope.Path = enclosingPath;
        }

        return metadata.Factory(values);
    }

    // No trim suppression here: FieldDescriptor.ClrType now carries
    // [DynamicallyAccessedMembers(PublicFields)], so the annotation the enum path needs reaches
    // WireValues.ReadValue from the descriptor itself.
    private static object? ReadField(
        FieldDescriptor field,
        object? raw,
        VersionableMetadata metadata,
        IReadOnlySet<string>? lazyFields,
        ReadScope scope)
    {
        if (lazyFields is not null && lazyFields.Contains(field.WireName))
        {
            // The backend's sentinel stands in for the value until something reads it; running it
            // through a converter here would defeat the point of not having loaded it.
            return raw;
        }

        object? validated = ValidateLiteral(field, raw, metadata);
        return field.WireReader is not null
            ? field.WireReader(validated)
            : WireValues.ReadValue(validated, field.ClrType, scope);
    }

    /// <summary>
    /// Produces the value of a field the backend deliberately did not read.
    /// </summary>
    /// <remarks>
    /// Python holds a <c>LazyArray</c> sentinel in the field and raises <c>ArrayNotLoadedError</c>
    /// when something reads it, so the failure lands on the access rather than the load. C# v1 has
    /// no equivalent: a lazy-instance proxy is Tier 3 and post-v1
    /// (<c>docs/plans/csharp-port.md</c>), and a <c>Tensor&lt;double&gt;</c> property cannot hold a
    /// sentinel. So the decision has to be made here, and there are only two honest answers:
    /// <list type="bullet">
    ///   <item>
    ///     <description>
    ///     The field <b>declares a default</b> — the schema has already said what an unpopulated
    ///     value looks like, so use it. A <c>MetadataOnly</c> load of such a type succeeds and
    ///     carries its array fields at their declared defaults.
    ///     </description>
    ///   </item>
    ///   <item>
    ///     <description>
    ///     The field <b>declares no default</b> — <see cref="Errors.ArrayNotLoadedException"/>,
    ///     which is the exception the option's own contract names. Inventing an empty tensor here
    ///     would be indistinguishable from a genuinely empty one, and the caller would have no way
    ///     to tell a skipped read from real data.
    ///     </description>
    ///   </item>
    /// </list>
    /// Neither answer is "missing from the file", which is the bug this replaced.
    /// </remarks>
    /// <param name="field">The field the backend skipped.</param>
    /// <param name="metadata">The declaring type's metadata.</param>
    /// <param name="path">The field's path from the load root, for the message.</param>
    /// <returns>The declared default.</returns>
    /// <exception cref="Errors.ArrayNotLoadedException">The field declares no default.</exception>
    private static object? SkippedFieldValue(FieldDescriptor field, VersionableMetadata metadata, string path)
    {
        if (TryDeclaredDefault(field, out object? declared))
        {
            return declared;
        }

        throw new ArrayNotLoadedException(
            $"Field '{path}' of '{metadata.Name}' was skipped by this load, not missing "
                + $"from the file: {nameof(VersionableLoadOptions.MetadataOnly)}, or a "
                + $"{nameof(VersionableLoadOptions.Preload)} set that leaves it out, told the backend "
                + $"not to read its data, and the property declares no default to stand in for it. "
                + $"Load with {nameof(VersionableLoadOptions.PreloadAll)}, name '{path}' in "
                + $"{nameof(VersionableLoadOptions.Preload)}, or give the property a default.");
    }

    /// <summary>
    /// Narrows a skip set to the entries belonging to one field's value.
    /// </summary>
    /// <remarks>
    /// Entries are <c>/</c>-separated <em>field-name</em> chains from the load root
    /// (<see cref="Backends.BackendLoadResult.LazyFields"/>), so descending into a field means
    /// keeping the entries under its name and dropping exactly that one segment — at every depth,
    /// with no arithmetic about how far down the value sits.
    /// <para>
    /// Container indexes never appear in a path, which is what makes one segment enough. A
    /// <c>Dictionary&lt;string, List&lt;Inner&gt;&gt;</c> named <c>groups</c> whose element type
    /// skips <c>samples</c> records <c>groups/samples</c>, so the narrowed set reaching every
    /// element — at any container depth — is <c>samples</c>, which is exactly what that element's
    /// materializer looks up. The alternative, recording an index per level, cannot work: a
    /// generated container reader hands the engine an element, never which element, so a path
    /// carrying <c>0</c> could never be matched back.
    /// </para>
    /// <para>
    /// The prefix match stays exact, which is what stops <c>a/b/values</c> from being read as a
    /// skip of <c>a</c>'s own <c>values</c>.
    /// </para>
    /// </remarks>
    /// <param name="lazyFields">The enclosing object's skips.</param>
    /// <param name="field">The field being read.</param>
    /// <returns>The skips belonging to that field's value, or <see langword="null"/> when none do.</returns>
    private static IReadOnlySet<string>? ScopeInto(IReadOnlySet<string>? lazyFields, FieldDescriptor field)
    {
        if (lazyFields is null || lazyFields.Count == 0)
        {
            return null;
        }

        string prefix = field.WireName + "/";
        HashSet<string>? scoped = null;

        foreach (string entry in lazyFields)
        {
            if (entry.StartsWith(prefix, StringComparison.Ordinal))
            {
                (scoped ??= new HashSet<string>(StringComparer.Ordinal)).Add(entry[prefix.Length..]);
            }
        }

        return scoped;
    }

    /// <summary>
    /// Produces the field's declared default, if it has one.
    /// </summary>
    /// <remarks>
    /// <see cref="FieldDescriptor.HasDefault"/> set with a null
    /// <see cref="FieldDescriptor.DefaultFactory"/> is a state the generator never emits — the
    /// contract pairs them — but the record cannot forbid it, so both callers
    /// (<see cref="SkippedFieldValue"/> and <see cref="DefaultFor"/>) treat it as "no default" and
    /// raise their own error rather than dereferencing null. One place, so the two cannot disagree
    /// about what an impossible descriptor means.
    /// </remarks>
    /// <param name="field">The field.</param>
    /// <param name="value">The default, when there is one.</param>
    /// <returns><see langword="true"/> when the field declares a usable default.</returns>
    private static bool TryDeclaredDefault(FieldDescriptor field, out object? value)
    {
        if (field.HasDefault && field.DefaultFactory is not null)
        {
            value = field.DefaultFactory();
            return true;
        }

        value = null;
        return false;
    }

    private static object? DefaultFor(FieldDescriptor field, VersionableMetadata metadata)
    {
        if (TryDeclaredDefault(field, out object? declared))
        {
            return declared;
        }

        throw new BackendException(
            $"Field '{field.WireName}' is missing from the file and has no default value. "
                + $"'{metadata.Name}' (version {metadata.Version}) requires it: either add the field "
                + $"to the file, give the property a default, or add a migration that supplies one.");
    }

    private static void ApplyUnknownFieldPolicy(IDictionary<string, object?> fields, VersionableMetadata metadata)
    {
        if (metadata.Unknown == UnknownFieldPolicy.Preserve)
        {
            // Nothing to do at materialization: the values stay in the dictionary the caller
            // holds, which is what a later save writes back out.
            return;
        }

        HashSet<string> declared = new(metadata.Fields.Count, StringComparer.Ordinal);
        foreach (FieldDescriptor field in metadata.Fields)
        {
            declared.Add(field.WireName);
        }

        List<string>? unknown = null;
        foreach (string key in fields.Keys)
        {
            if (!declared.Contains(key))
            {
                (unknown ??= []).Add(key);
            }
        }

        if (unknown is null)
        {
            return;
        }

        unknown.Sort(StringComparer.Ordinal);

        if (metadata.Unknown == UnknownFieldPolicy.Error)
        {
            throw new UnknownFieldException(
                $"Unknown fields in data for {metadata.Name}: [{string.Join(", ", unknown)}].");
        }

        foreach (string key in unknown)
        {
            fields.Remove(key);
        }
    }

    /// <summary>
    /// Checks a field's value against <see cref="FieldDescriptor.LiteralOptions"/>, substituting
    /// the fallback or failing.
    /// </summary>
    /// <remarks>
    /// <b>Whole-field only, by construction.</b> <see cref="FieldDescriptor"/> has one options list
    /// per field and nothing that says where in the field's type the literal sits, so a
    /// <c>list[Literal['fast', 'slow']]</c> cannot be validated here — the value reaching this
    /// method is the list, and no list is ever equal to <c>'fast'</c>. The generator must therefore
    /// leave <see cref="FieldDescriptor.LiteralOptions"/> <see langword="null"/> for a container of
    /// literals and validate per element inside the <see cref="FieldDescriptor.WireReader"/> it
    /// already emits for that field; setting both would reject every such file. Python has no
    /// equivalent split because it walks the annotation at load time and meets the
    /// <c>Literal</c> at whatever depth it sits.
    /// <para>
    /// Giving <see cref="FieldDescriptor"/> a depth or path member would let the engine do it
    /// instead; that is a contract amendment, reported with task 2d rather than assumed here.
    /// </para>
    /// </remarks>
    /// <param name="field">The field being read.</param>
    /// <param name="raw">The wire value, before materialization.</param>
    /// <param name="metadata">The declaring type's metadata.</param>
    /// <returns>The value to materialize: <paramref name="raw"/>, or the declared fallback.</returns>
    private static object? ValidateLiteral(FieldDescriptor field, object? raw, VersionableMetadata metadata)
    {
        if (!metadata.ValidateLiterals || field.LiteralOptions is null)
        {
            return raw;
        }

        foreach (object? option in field.LiteralOptions)
        {
            if (LiteralEquals(option, raw))
            {
                return raw;
            }
        }

        if (field.HasLiteralFallback)
        {
            VersionableLog.Warn(
                $"{metadata.Name}.{field.WireName}: value {WireValues.Quote(raw)} is not a declared "
                    + $"literal option. Using the fallback {WireValues.Quote(field.LiteralFallback)}.");
            return field.LiteralFallback;
        }

        throw new ConverterException(
            $"{metadata.Name}.{field.WireName}: value {WireValues.Quote(raw)} is not a valid Literal "
                + $"option. Allowed values: [{string.Join(", ", field.LiteralOptions.Select(WireValues.Quote))}].");
    }

    /// <summary>
    /// Compares a declared literal option with a value read from a file.
    /// </summary>
    /// <remarks>
    /// Deliberately stricter than Python on one point: Python's <c>in</c> uses <c>==</c>, under
    /// which <c>True == 1</c>, so a <c>Literal[1]</c> field there accepts <c>true</c> from a file.
    /// Matching that would mean accepting a boolean into an integer field, which no C# schema can
    /// represent anyway. Everything else is the same comparison, with integral widths normalized
    /// because a JSON reader is free to hand back <c>long</c> where the option was written
    /// <c>int</c>.
    /// </remarks>
    private static bool LiteralEquals(object? option, object? value)
    {
        if (option is null || value is null)
        {
            return option is null && value is null;
        }

        if (option is string optionText)
        {
            return value is string valueText && string.Equals(optionText, valueText, StringComparison.Ordinal);
        }

        if (option is bool optionFlag)
        {
            return value is bool valueFlag && optionFlag == valueFlag;
        }

        if (IsIntegral(option) && IsIntegral(value))
        {
            return Convert.ToInt64(option, CultureInfo.InvariantCulture)
                == Convert.ToInt64(value, CultureInfo.InvariantCulture);
        }

        return option.Equals(value);
    }

    private static bool IsIntegral(object value) =>
        value is sbyte or byte or short or ushort or int or uint or long or ulong;

    private static VersionableMetadata ResolveConcrete(EnvelopeMetadata envelope, VersionableMetadata declared)
    {
        string? name = envelope.ObjectName;
        if (string.IsNullOrEmpty(name) || string.Equals(name, declared.Name, StringComparison.Ordinal))
        {
            // No envelope, or the declared identity. Also covers Register = false types, which
            // are absent from the name index and could not be resolved through it.
            return declared;
        }

        if (declared.OldNames is not null && declared.OldNames.Contains(name, StringComparer.Ordinal))
        {
            return declared;
        }

        if (!VersionableRegistry.TryGetByName(name, out VersionableMetadata? found))
        {
            throw new BackendException(
                $"Unknown nested object type '{name}' (declared field type: {declared.ClrType}). "
                    + "The type is not registered, or it has been removed.");
        }

        if (!declared.ClrType.IsAssignableFrom(found.ClrType))
        {
            throw new BackendException(
                $"Nested object type '{name}' resolves to {found.ClrType}, which is not assignable "
                    + $"to the declared field type {declared.ClrType}.");
        }

        return found;
    }
}
