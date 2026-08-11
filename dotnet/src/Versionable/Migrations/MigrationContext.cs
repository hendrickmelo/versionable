namespace Versionable.Migrations;

/// <summary>
/// The raw field dictionary an imperative migration reads and rewrites.
/// </summary>
/// <remarks>
/// Python counterpart: <c>MigrationContext</c> in <c>src/versionable/_migration.py</c>, member for
/// member. Values are wire values — what the backend read out of the file, before any converter or
/// materialization — because a migration renames and drops keys that the current schema has no
/// CLR type for.
/// <para>
/// A view over the load's own dictionary, not a copy: every mutation lands in the dictionary the
/// chain returns.
/// </para>
/// </remarks>
public sealed class MigrationContext
{
    private readonly IDictionary<string, object?> _fields;

    /// <summary>Initializes a new instance of the <see cref="MigrationContext"/> class.</summary>
    /// <param name="fields">The field dictionary to wrap. Mutated in place.</param>
    public MigrationContext(IDictionary<string, object?> fields)
    {
        ArgumentNullException.ThrowIfNull(fields);
        _fields = fields;
    }

    /// <summary>The wire names currently present.</summary>
    /// <remarks>
    /// A live view, so removing a field while enumerating it throws — take
    /// <see cref="ToDictionary"/> first when the migration decides what to drop by walking the keys.
    /// </remarks>
    public ICollection<string> Keys => _fields.Keys;

    /// <summary>Reads or writes one field.</summary>
    /// <param name="key">Wire name.</param>
    /// <returns>The value.</returns>
    /// <exception cref="KeyNotFoundException">
    /// Reading a field the file does not carry, as Python's <c>ctx[key]</c> raises <c>KeyError</c>.
    /// </exception>
    public object? this[string key]
    {
        get => _fields[key];
        set => _fields[key] = value;
    }

    /// <summary>Whether a field is present.</summary>
    /// <param name="key">Wire name.</param>
    /// <returns><see langword="true"/> when the file carries it.</returns>
    public bool Contains(string key) => _fields.ContainsKey(key);

    /// <summary>Reads a field without throwing when it is absent.</summary>
    /// <param name="key">Wire name.</param>
    /// <param name="value">The value, or <see langword="null"/> when absent.</param>
    /// <returns><see langword="true"/> when the field was present.</returns>
    public bool TryGetValue(string key, out object? value) => _fields.TryGetValue(key, out value);

    /// <summary>Removes a field and returns its value.</summary>
    /// <param name="key">Wire name.</param>
    /// <returns>The removed value.</returns>
    /// <exception cref="KeyNotFoundException">The field is absent, as Python's <c>pop(key)</c> raises.</exception>
    public object? Pop(string key)
    {
        if (!_fields.TryGetValue(key, out object? value))
        {
            throw new KeyNotFoundException($"The migrating file has no field '{key}'.");
        }

        _fields.Remove(key);
        return value;
    }

    /// <summary>Removes a field, falling back to a value when it is absent.</summary>
    /// <param name="key">Wire name.</param>
    /// <param name="fallback">Value to return when the field is absent.</param>
    /// <returns>The removed value, or <paramref name="fallback"/>.</returns>
    public object? Pop(string key, object? fallback)
    {
        if (!_fields.TryGetValue(key, out object? value))
        {
            return fallback;
        }

        _fields.Remove(key);
        return value;
    }

    /// <summary>Removes a field, ignoring its absence.</summary>
    /// <param name="key">Wire name.</param>
    public void Drop(string key) => _fields.Remove(key);

    /// <summary>Copies the fields out.</summary>
    /// <returns>A new dictionary holding the current fields.</returns>
    /// <remarks>Python counterpart: <c>toDict()</c>. A snapshot; writing to it changes nothing.</remarks>
    public Dictionary<string, object?> ToDictionary() => new(_fields, StringComparer.Ordinal);
}
