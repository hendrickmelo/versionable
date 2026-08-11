using Versionable.Backends;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The two commenting backends against one fixture: <c>commentDefaults</c> has to mean the same
/// thing in YAML as it does in TOML.
/// </summary>
/// <remarks>
/// Two formats, two emitters, two independent implementations of "is this field still at its
/// default" — <c>YamlBackend.IsAtDefault</c> compares rendered blocks, <c>TomlDefaults</c> compares
/// lowered wire values — and nothing but a test written across both makes them answer alike. The
/// case that matters is the nested one: a defaulted nested object is several lines in either
/// format, and the two backends could plausibly have disagreed about whether its header and
/// envelope are part of the field or part of the file.
/// <para>
/// The rule both settle on, and the deliberate divergences from Python behind it, are recorded on
/// <see cref="BackendSaveOptions.CommentDefaults"/>. Python's own two backends do not agree with
/// each other here, which is the reason this file exists.
/// </para>
/// </remarks>
[Collection(RegistryCollection.Name)]
public class YamlTomlCommentDefaultsTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"versionable-comment-defaults-{Path.GetRandomFileName()}");

    public YamlTomlCommentDefaultsTests()
    {
        GoldenSchemas.EnsureRegistered();
        foreach (VersionableMetadata metadata in TomlCommentSchemas.All)
        {
            VersionableRegistry.Register(metadata);
        }

        Directory.CreateDirectory(_directory);
    }

    public void Dispose()
    {
        Directory.Delete(_directory, recursive: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Both backends comment out exactly the fields at their default and leave the rest — and the
    /// envelope — live.
    /// </summary>
    /// <param name="extension">The format under test.</param>
    [Theory]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    public void both_backends_comment_the_same_flat_fields(string extension)
    {
        // `name` is at its default, `port` is not, `required` has none.
        string text = SaveWithComments(new TomlSettings("r", port: 9000), $"settings{extension}");

        Assert.Equal(["name"], CommentedKeys(text));
        Assert.DoesNotContain("# object", text, StringComparison.Ordinal);
    }

    /// <summary>
    /// A defaulted nested object is commented out whole in both formats: its own key, its fields,
    /// and the envelope it would have carried.
    /// </summary>
    /// <remarks>
    /// The reconciliation this file exists for. Neither backend leaves a live header behind for an
    /// object the file does not set — which is what Python's TOML backend does, and what would
    /// otherwise have made the two C# backends disagree.
    /// </remarks>
    /// <param name="extension">The format under test.</param>
    [Theory]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    public void both_backends_comment_a_defaulted_nested_object_whole(string extension)
    {
        string text = SaveWithComments(new TomlHostRoot("x"), $"nested{extension}");

        // The field's key is commented, whatever the format spells it as.
        Assert.Equal(["db"], CommentedKeys(text));

        // And so is everything the block contains, envelope included: no line of it is live.
        Assert.DoesNotContain("localhost", Live(text), StringComparison.Ordinal);
        Assert.DoesNotContain("5432", Live(text), StringComparison.Ordinal);
        Assert.DoesNotContain("TomlHost\"", Live(text), StringComparison.Ordinal);
        Assert.DoesNotContain("TomlHost\n", Live(text), StringComparison.Ordinal);

        // The root's own envelope is untouched, so the file still says what it is.
        Assert.Contains("TomlHostRoot", Live(text), StringComparison.Ordinal);
    }

    /// <summary>
    /// Uncommenting a defaulted nested object never costs a field that was set — in either format,
    /// whichever is declared first.
    /// </summary>
    /// <remarks>
    /// The semantic the two backends have to share, stated as an outcome rather than a layout
    /// because they reach it differently. YAML's comments are indentation-scoped, so a commented
    /// block is inert wherever it sits and the fields stay in declaration order. TOML's are not: a
    /// bare key binds to the most recent table header, so a live <c>name = "prod"</c> printed after
    /// a commented <c>[db]</c> block would join <c>[db.__versionable__]</c> the moment the block is
    /// uncommented and be dropped as an unknown envelope key. TOML therefore emits commented
    /// sections with the other sections; see
    /// <see cref="TomlCommentDefaultsTests.a_commented_section_never_precedes_a_live_key_line"/>.
    /// <para>
    /// Asserted by performing the edit the comment invites — strip every <c>#</c>, reload — which
    /// is the only form of the claim that holds both backends to the same standard.
    /// </para>
    /// </remarks>
    /// <param name="extension">The format under test.</param>
    [Theory]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    public void uncommenting_a_defaulted_nested_object_keeps_the_fields_that_were_set(string extension)
    {
        // `db` is declared before `name`, is at its default, and is therefore commented whole.
        string text = SaveWithComments(new TomlOrderRoot(name: "prod"), $"order{extension}");
        Assert.Equal(["db"], CommentedKeys(text));

        string path = Path.Combine(_directory, $"uncommented{extension}");
        File.WriteAllText(path, text.Replace("# ", string.Empty, StringComparison.Ordinal));

        TomlOrderRoot loaded = VersionableFile.Load<TomlOrderRoot>(path);
        Assert.Equal("prod", loaded.Name);
        Assert.Equal("localhost", loaded.Db.Host);
        Assert.Equal(5432, loaded.Db.Port);
    }

    /// <summary>
    /// The commenting is cosmetic in both formats: what loads back is what was saved.
    /// </summary>
    /// <param name="extension">The format under test.</param>
    [Theory]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    public void a_commented_file_loads_back_identically_in_both_formats(string extension)
    {
        string path = Path.Combine(_directory, $"nested{extension}");
        VersionableFile.Save(
            new TomlHostRoot("x"),
            path,
            options: new BackendSaveOptions { CommentDefaults = true });

        TomlHostRoot loaded = VersionableFile.Load<TomlHostRoot>(path);

        Assert.Equal("x", loaded.Label);
        Assert.Equal("localhost", loaded.Db.Host);
        Assert.Equal(5432, loaded.Db.Port);
    }

    /// <summary>
    /// The top-level keys a file comments out, format-independently.
    /// </summary>
    /// <remarks>
    /// A commented field is a run of <c>#</c> lines whose first one names it — <c>db:</c> in YAML,
    /// <c>[db]</c> in TOML — so taking the first token of each comment run and keeping the ones
    /// that are field names of the fixture gives the same answer for either format. Anything
    /// deeper in the run belongs to the field the run opened.
    /// </remarks>
    /// <param name="text">A file written with <c>commentDefaults</c>.</param>
    /// <returns>The commented field names, in the order they appear.</returns>
    private static IReadOnlyList<string> CommentedKeys(string text)
    {
        List<string> keys = [];
        bool inRun = false;
        foreach (string line in text.ReplaceLineEndings("\n").Split('\n'))
        {
            if (!line.StartsWith("# ", StringComparison.Ordinal))
            {
                inRun = false;
                continue;
            }

            if (!inRun)
            {
                keys.Add(line[2..].TrimStart('[').Split([':', '=', ']'])[0].Trim());
            }

            inRun = true;
        }

        return keys;
    }

    /// <summary>Everything the file says once the comments are struck out.</summary>
    /// <param name="text">A file written with <c>commentDefaults</c>.</param>
    /// <returns>The live lines, newline-joined.</returns>
    private static string Live(string text) =>
        string.Join(
            '\n',
            text.ReplaceLineEndings("\n")
                .Split('\n')
                .Where(line => !line.TrimStart().StartsWith('#')));

    private string SaveWithComments(object value, string file)
    {
        string path = Path.Combine(_directory, file);
        VersionableFile.Save(value, path, options: new BackendSaveOptions { CommentDefaults = true });
        return File.ReadAllText(path);
    }
}
