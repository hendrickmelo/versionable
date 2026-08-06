namespace Versionable.Migrations;

/// <summary>
/// One operation of a declarative <see cref="Migration"/>, held as data rather than as an opaque
/// closure so a chain can be read back out of a compiled assembly.
/// </summary>
/// <remarks>
/// Python counterpart: the frozen dataclasses <c>_RenameOp</c> … <c>_RequiresUpgradeOp</c> in
/// <c>src/versionable/_migration.py</c>. The hierarchy is closed — the base constructor is
/// <c>private protected</c> — because <see cref="Migration.Apply"/> switches over every case, and
/// an op it did not recognise would surface as a load-time failure on a file that was fine.
/// <para>
/// The four ops Python spells with a callable — convert, derive, split, merge — carry delegates
/// here too. Every other op carries only values, exactly as Python's do, which is what lets a
/// Python migration be re-emitted as C# source rather than reimplemented by hand.
/// </para>
/// </remarks>
public abstract record MigrationOp
{
    private protected MigrationOp()
    {
    }
}

/// <summary>Moves a value from one wire name to another.</summary>
/// <param name="OldName">Wire name the old file used.</param>
/// <param name="NewName">Wire name the new schema uses.</param>
/// <remarks>A file without <paramref name="OldName"/> is left alone, as in Python.</remarks>
public sealed record RenameOp(string OldName, string NewName) : MigrationOp;

/// <summary>Introduces a field that older files do not carry.</summary>
/// <param name="Field">Wire name to add.</param>
/// <param name="Value">Value to add it with, when <paramref name="Factory"/> is null.</param>
/// <param name="Factory">
/// Produces the value per application, for defaults that must not be shared between files.
/// Python counterpart: passing a callable as <c>default</c>.
/// </param>
/// <remarks>
/// Never overwrites a value already in the file: the point of the op is to say what the field
/// meant before it existed, which is frequently not the current schema default.
/// </remarks>
public sealed record AddOp(string Field, object? Value, Func<object?>? Factory) : MigrationOp;

/// <summary>Removes a field the schema no longer has.</summary>
/// <param name="Field">Wire name to remove. Absence is not an error.</param>
public sealed record DropOp(string Field) : MigrationOp;

/// <summary>Rewrites a field's value in place.</summary>
/// <param name="Field">Wire name to convert. Absence is not an error.</param>
/// <param name="Via">Maps the old value to the new one.</param>
/// <param name="Reverse">
/// The inverse, for a future downgrade path. Recorded but not run — Python records it the same
/// way and never calls it either.
/// </param>
public sealed record ConvertOp(
    string Field,
    Func<object?, object?> Via,
    Func<object?, object?>? Reverse) : MigrationOp;

/// <summary>Computes a new field from an existing one, leaving the source in place.</summary>
/// <param name="Field">Wire name to write.</param>
/// <param name="FromField">Wire name to read. When absent, nothing is written.</param>
/// <param name="Via">Maps the source value to the derived one.</param>
public sealed record DeriveOp(string Field, string FromField, Func<object?, object?> Via) : MigrationOp;

/// <summary>Replaces one field with several derived from it.</summary>
/// <param name="Field">Wire name to consume. When absent, nothing happens.</param>
/// <param name="Targets">Fields to write, in order.</param>
public sealed record SplitOp(string Field, IReadOnlyList<SplitTarget> Targets) : MigrationOp;

/// <summary>One output of a <see cref="SplitOp"/>.</summary>
/// <param name="Field">Wire name to write.</param>
/// <param name="Via">Maps the source value to this field's value.</param>
public sealed record SplitTarget(string Field, Func<object?, object?> Via);

/// <summary>Replaces several fields with one combined from them.</summary>
/// <param name="Fields">Wire names to consume, in the order <paramref name="Via"/> expects.</param>
/// <param name="Into">Wire name to write.</param>
/// <param name="Via">
/// Combines the values of the fields that were present. Fields absent from the file are absent
/// from the array too — Python's <c>via(*values)</c> is applied to the present values only — so a
/// combiner that indexes blindly must be prepared for a short array.
/// </param>
public sealed record MergeOp(
    IReadOnlyList<string> Fields,
    string Into,
    Func<object?[], object?> Via) : MigrationOp;

/// <summary>
/// Declares that the migration cannot be applied to a file the caller only wants to read.
/// </summary>
/// <remarks>
/// Python counterpart: <c>Migration.requiresUpgrade()</c>. Applying the migration without
/// <c>upgradeInPlace</c> throws <see cref="Errors.UpgradeRequiredException"/>; the op is a
/// declaration, not a rewrite, so it changes no field itself.
/// </remarks>
public sealed record RequiresUpgradeOp : MigrationOp;
