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

        return Materialize(fields, actual, lazyFields: null, scope);
    }

    /// <summary>Builds an instance from already-migrated field values.</summary>
    /// <param name="fields">Raw field values keyed by wire name, envelope keys already stripped.</param>
    /// <param name="metadata">Metadata of the type to build.</param>
    /// <param name="lazyFields">
    /// Wire names the backend deliberately left unmaterialized, whose values pass straight
    /// through to the instance as the backend's own sentinels.
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
        for (int index = 0; index < metadata.Fields.Count; index++)
        {
            FieldDescriptor field = metadata.Fields[index];
            values[index] = fields.TryGetValue(field.WireName, out object? raw)
                ? ReadField(field, raw, metadata, lazyFields, scope)
                : DefaultFor(field, metadata);
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

    private static object? DefaultFor(FieldDescriptor field, VersionableMetadata metadata)
    {
        if (field.HasDefault && field.DefaultFactory is not null)
        {
            return field.DefaultFactory();
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
