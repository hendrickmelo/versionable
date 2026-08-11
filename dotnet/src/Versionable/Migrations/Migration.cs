using Versionable.Errors;

namespace Versionable.Migrations;

/// <summary>
/// A declarative migration: an ordered list of <see cref="MigrationOp"/> that reshapes the raw
/// field dictionary of one schema version into the next.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>Migration</c> builder in <c>src/versionable/_migration.py</c>, op
/// for op. Declared as a member of the nested <c>Migrate</c> class named for the version it reads:
/// <code>
/// public static class Migrate
/// {
///     public static readonly Migration V1 = new Migration().Rename("title", "name").Drop("debug");
///     public static readonly Migration V2 = new Migration().Add("timeout_ms", 0);
/// }
/// </code>
/// <para>
/// <b>Each builder call returns a new instance</b> rather than mutating the receiver, which
/// Python's does. The op semantics are identical; what differs is that a <c>Migration</c> published
/// as a <c>static readonly</c> field — the way every declared migration is published — cannot be
/// extended by one caller into something a later caller sees. A partly built migration that is
/// never assigned is simply dropped, whereas Python's would silently keep the ops.
/// </para>
/// </remarks>
public sealed class Migration
{
    private static readonly MigrationOp[] _empty = [];

    private readonly MigrationOp[] _ops;

    /// <summary>Initializes a new instance of the <see cref="Migration"/> class with no operations.</summary>
    public Migration()
        : this(_empty)
    {
    }

    private Migration(MigrationOp[] ops)
    {
        _ops = ops;
        Ops = Array.AsReadOnly(ops);
    }

    /// <summary>The operations, in the order they were declared and are applied.</summary>
    /// <remarks>
    /// A read-only wrapper rather than the backing array, which a caller could otherwise cast
    /// back to <c>MigrationOp[]</c> and rewrite. Python hands back a copy for the same reason.
    /// </remarks>
    public IReadOnlyList<MigrationOp> Ops { get; }

    /// <summary>Moves a value from one wire name to another.</summary>
    /// <param name="oldName">Wire name the old file used.</param>
    /// <param name="newName">Wire name the current schema uses.</param>
    /// <returns>A migration with the operation appended.</returns>
    public Migration Rename(string oldName, string newName)
    {
        ArgumentNullException.ThrowIfNull(oldName);
        ArgumentNullException.ThrowIfNull(newName);
        return With(new RenameOp(oldName, newName));
    }

    /// <summary>Adds a field that files at the older version do not carry.</summary>
    /// <param name="field">Wire name to add.</param>
    /// <param name="value">Value to give it. A field already present keeps its own value.</param>
    /// <returns>A migration with the operation appended.</returns>
    /// <remarks>
    /// The value is stored as-is, delegate or not: a delegate passed here becomes the field's
    /// value rather than being called, which no backend can write. Use
    /// <see cref="AddComputed(string, Func{object})"/> for a value computed per application —
    /// Python's <c>add(field, default=callable)</c>.
    /// </remarks>
    public Migration Add(string field, object? value)
    {
        ArgumentNullException.ThrowIfNull(field);
        return With(new AddOp(field, value, null));
    }

    /// <summary>Adds a field whose value is produced per application.</summary>
    /// <param name="field">Wire name to add.</param>
    /// <param name="defaultFactory">Produces the value each time the migration runs.</param>
    /// <returns>A migration with the operation appended.</returns>
    /// <remarks>
    /// Python counterpart: <c>add(field, default=list)</c> — a callable default, which Python calls
    /// rather than stores. Spelled as its own method because C# 10 gave lambdas a natural
    /// conversion to <see cref="object"/>, so an overload of <see cref="Add(string, object?)"/>
    /// would be ambiguous at every call site.
    /// </remarks>
    public Migration AddComputed(string field, Func<object?> defaultFactory)
    {
        ArgumentNullException.ThrowIfNull(field);
        ArgumentNullException.ThrowIfNull(defaultFactory);
        return With(new AddOp(field, null, defaultFactory));
    }

    /// <summary>Removes a field the current schema no longer has.</summary>
    /// <param name="field">Wire name to remove. Absence is not an error.</param>
    /// <returns>A migration with the operation appended.</returns>
    public Migration Drop(string field)
    {
        ArgumentNullException.ThrowIfNull(field);
        return With(new DropOp(field));
    }

    /// <summary>Rewrites a field's value in place.</summary>
    /// <param name="field">Wire name to convert. Absence is not an error.</param>
    /// <param name="via">Maps the old value to the new one.</param>
    /// <param name="reverse">The inverse, recorded for a future downgrade path and never run.</param>
    /// <returns>A migration with the operation appended.</returns>
    public Migration Convert(
        string field,
        Func<object?, object?> via,
        Func<object?, object?>? reverse = null)
    {
        ArgumentNullException.ThrowIfNull(field);
        ArgumentNullException.ThrowIfNull(via);
        return With(new ConvertOp(field, via, reverse));
    }

    /// <summary>Computes a new field from an existing one, which stays in place.</summary>
    /// <param name="field">Wire name to write.</param>
    /// <param name="fromField">Wire name to read. When absent, nothing is written.</param>
    /// <param name="via">Maps the source value to the derived one.</param>
    /// <returns>A migration with the operation appended.</returns>
    public Migration Derive(string field, string fromField, Func<object?, object?> via)
    {
        ArgumentNullException.ThrowIfNull(field);
        ArgumentNullException.ThrowIfNull(fromField);
        ArgumentNullException.ThrowIfNull(via);
        return With(new DeriveOp(field, fromField, via));
    }

    /// <summary>Replaces one field with several derived from it.</summary>
    /// <param name="field">Wire name to consume. When absent, nothing happens.</param>
    /// <param name="targets">Fields to write, in order.</param>
    /// <returns>A migration with the operation appended.</returns>
    /// <remarks>
    /// Python passes a dict, whose insertion order it preserves; the targets are an ordered list
    /// here because a C# dictionary's enumeration order is not part of its contract, and the phase-6
    /// converter has to emit the same order it read.
    /// </remarks>
    public Migration Split(string field, params SplitTarget[] targets)
    {
        ArgumentNullException.ThrowIfNull(field);
        ArgumentNullException.ThrowIfNull(targets);
        return With(new SplitOp(field, (SplitTarget[])targets.Clone()));
    }

    /// <summary>Replaces several fields with one combined from them.</summary>
    /// <param name="fields">Wire names to consume, in the order <paramref name="via"/> expects.</param>
    /// <param name="into">Wire name to write.</param>
    /// <param name="via">Combines the values of the fields that were present.</param>
    /// <returns>A migration with the operation appended.</returns>
    public Migration Merge(IReadOnlyList<string> fields, string into, Func<object?[], object?> via)
    {
        ArgumentNullException.ThrowIfNull(fields);
        ArgumentNullException.ThrowIfNull(into);
        ArgumentNullException.ThrowIfNull(via);
        return With(new MergeOp([.. fields], into, via));
    }

    /// <summary>Declares that this migration rewrites the file rather than only the loaded data.</summary>
    /// <returns>A migration with the operation appended.</returns>
    public Migration RequiresUpgrade() => With(new RequiresUpgradeOp());

    /// <summary>Concatenates another migration's operations after this one's.</summary>
    /// <param name="other">The migration to run second.</param>
    /// <returns>A migration holding both op lists, leaving both operands unchanged.</returns>
    public Migration Then(Migration other)
    {
        ArgumentNullException.ThrowIfNull(other);

        MigrationOp[] combined = new MigrationOp[_ops.Length + other._ops.Length];
        Array.Copy(_ops, combined, _ops.Length);
        Array.Copy(other._ops, 0, combined, _ops.Length, other._ops.Length);
        return new Migration(combined);
    }

    /// <summary>Runs every operation over <paramref name="fields"/>, in declaration order.</summary>
    /// <param name="fields">
    /// Raw field values keyed by wire name, envelope keys already stripped. Mutated in place and
    /// returned: unlike Python's <c>applyMigrations</c>, which copies the dict once per step, the
    /// dictionary handed to a chain is built by the load path for that one load and has no other
    /// reader.
    /// </param>
    /// <param name="upgradeInPlace">Whether the caller has permitted file-rewriting migrations.</param>
    /// <returns><paramref name="fields"/>, reshaped.</returns>
    /// <exception cref="UpgradeRequiredException">
    /// The migration declares <see cref="RequiresUpgrade"/> and <paramref name="upgradeInPlace"/>
    /// is <see langword="false"/>.
    /// </exception>
    public IDictionary<string, object?> Apply(IDictionary<string, object?> fields, bool upgradeInPlace = false)
    {
        ArgumentNullException.ThrowIfNull(fields);

        foreach (MigrationOp op in _ops)
        {
            ApplyOp(op, fields, upgradeInPlace);
        }

        return fields;
    }

    private static void ApplyOp(MigrationOp op, IDictionary<string, object?> fields, bool upgradeInPlace)
    {
        switch (op)
        {
            case RenameOp rename:
                if (fields.TryGetValue(rename.OldName, out object? renamed))
                {
                    fields.Remove(rename.OldName);
                    fields[rename.NewName] = renamed;
                }

                break;

            case AddOp add:
                if (!fields.ContainsKey(add.Field))
                {
                    fields[add.Field] = add.Factory is null ? add.Value : add.Factory();
                }

                break;

            case DropOp drop:
                fields.Remove(drop.Field);
                break;

            case ConvertOp convert:
                if (fields.TryGetValue(convert.Field, out object? converting))
                {
                    fields[convert.Field] = convert.Via(converting);
                }

                break;

            case DeriveOp derive:
                if (fields.TryGetValue(derive.FromField, out object? source))
                {
                    fields[derive.Field] = derive.Via(source);
                }

                break;

            case SplitOp split:
                if (fields.TryGetValue(split.Field, out object? splitting))
                {
                    fields.Remove(split.Field);
                    foreach (SplitTarget target in split.Targets)
                    {
                        fields[target.Field] = target.Via(splitting);
                    }
                }

                break;

            case MergeOp merge:
                ApplyMerge(merge, fields);
                break;

            case RequiresUpgradeOp when !upgradeInPlace:
                throw new UpgradeRequiredException(
                    "This migration requires in-place file modification. Load with "
                        + "VersionableLoadOptions { UpgradeInPlace = true } to permit it.");

            case RequiresUpgradeOp:
                break;

            default:
                throw new MigrationException($"Unknown migration operation: {op.GetType().Name}.");
        }
    }

    private static void ApplyMerge(MergeOp merge, IDictionary<string, object?> fields)
    {
        // Python collects only the fields that are present and passes them positionally, so a file
        // missing one of them merges the rest rather than failing. Same here, short array and all.
        List<object?> present = new(merge.Fields.Count);
        foreach (string field in merge.Fields)
        {
            if (fields.TryGetValue(field, out object? value))
            {
                fields.Remove(field);
                present.Add(value);
            }
        }

        if (present.Count > 0)
        {
            fields[merge.Into] = merge.Via([.. present]);
        }
    }

    private Migration With(MigrationOp op)
    {
        MigrationOp[] extended = new MigrationOp[_ops.Length + 1];
        Array.Copy(_ops, extended, _ops.Length);
        extended[_ops.Length] = op;
        return new Migration(extended);
    }
}
