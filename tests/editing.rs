#![cfg(feature = "span")]

//! Located, trivia-preserving edit API tests.
//!
//! Every test here is independently runnable, e.g.
//! `cargo test --locked --all-features --test editing raw_string`.
//!
//! The `.kdl` inputs live in `tests/edit_fixtures/` and can be inspected (or
//! fed to other tooling) directly: `raw_string.kdl`, `comments.kdl`,
//! `multiline_value.kdl`, `crlf.kdl`.

use kdl::edit::{EditKind, EditPlan, SourceEdit, SourceVersion};
use kdl::{KdlDocument, KdlIdentifier, KdlValue};
use std::fs;
use std::path::Path;

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/edit_fixtures")
        .join(format!("{name}.kdl"));
    fs::read_to_string(path).expect("fixture must exist")
}

fn parse(src: &str) -> KdlDocument {
    KdlDocument::parse(src).expect("fixture must parse")
}

fn canonical(src: &str) -> KdlDocument {
    let mut doc = parse(src);
    doc.clear_format_recursive();
    doc
}

/// Asserts that every source byte outside the edited ranges survives, in
/// order and contiguous, with only the declared replacements inserted.
fn assert_only_ranges_changed(src: &str, edits: &[SourceEdit], out: &str) {
    assert!(
        edits.windows(2).all(|w| w[0].end() <= w[1].start()),
        "fixture edits must not overlap"
    );
    let mut skeleton = String::new();
    let mut cursor = 0usize;
    for edit in edits {
        skeleton.push_str(&src[cursor..edit.start()]);
        cursor = edit.end();
    }
    skeleton.push_str(&src[cursor..]);

    let mut stripped = String::new();
    let mut src_cursor = 0usize;
    let mut out_cursor = 0usize;
    for edit in edits {
        let gap = edit.start() - src_cursor;
        assert!(out.is_char_boundary(out_cursor + gap));
        stripped.push_str(&out[out_cursor..out_cursor + gap]);
        out_cursor += gap;
        assert!(out.is_char_boundary(out_cursor + edit.replacement.len()));
        assert_eq!(
            &out[out_cursor..out_cursor + edit.replacement.len()],
            edit.replacement
        );
        out_cursor += edit.replacement.len();
        src_cursor = edit.end();
    }
    stripped.push_str(&out[out_cursor..]);
    assert_eq!(
        stripped, skeleton,
        "untouched bytes must be preserved exactly"
    );
}

/// Full equivalence: applying edits and performing the same mutation through
/// the tree API must produce equal (canonicalized) ASTs, and the edited text
/// must parse successfully.
fn assert_tree_equiv(
    src: &str,
    edits: &[SourceEdit],
    out: &str,
    mutate: impl FnOnce(&mut KdlDocument),
) {
    assert!(
        KdlDocument::parse(out).is_ok(),
        "edited text must parse: {out:?}"
    );
    let edited = canonical(out);
    let mut oracle = parse(src);
    mutate(&mut oracle);
    oracle.clear_format_recursive();
    assert_eq!(
        edited, oracle,
        "located edits must match the tree-API mutation"
    );
    assert_only_ranges_changed(src, edits, out);
}

// --- raw_string.kdl ---------------------------------------------------------

#[test]
fn raw_string_preserves_other_literals() {
    let src = fixture("raw_string");
    let doc = parse(&src);
    let version = SourceVersion::new(src.clone());
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    let package = doc.get("package").unwrap();
    plan.set_argument(package, 0, KdlValue::from("10.0.0.1"))
        .unwrap();
    plan.set_property(package, "name", KdlValue::from("renamed"))
        .unwrap();
    let server = doc.get("server").unwrap();
    plan.set_property(server, "host", KdlValue::from("10.1.0.1"))
        .unwrap();

    let edits = plan.edits().to_vec();
    let out = plan.apply().unwrap();

    assert_tree_equiv(&src, &edits, &out, |doc| {
        doc.get_mut("package")
            .unwrap()
            .entry_mut(0)
            .unwrap()
            .set_value(KdlValue::from("10.0.0.1"));
        doc.get_mut("package").unwrap()["name"] = KdlValue::from("renamed");
        doc.get_mut("server").unwrap()["host"] = KdlValue::from("10.1.0.1");
    });

    // Raw literals that were not targeted keep their exact spelling.
    assert!(out.contains("path=#\"C:\\temp\\kdl\"#"));
    assert!(out.contains("greeting=##\"a \"raw\" word needs hashes\"##"));
    assert!(out.contains("raw=#\"untouched \"value\" here\"#"));

    // The edited property is a minimal replacement of the raw literal only.
    assert!(out.contains("name=renamed"));
    assert!(out.contains("host=\"10.1.0.1\""));
    assert!(out.contains("package \"10.0.0.1\""));
    assert_eq!(edits.len(), 3);
    assert_eq!(edits[0].kind(), EditKind::SetArgument);
    assert_eq!(edits[1].kind(), EditKind::SetProperty);
    assert_eq!(edits[2].kind(), EditKind::SetProperty);
}

#[test]
fn raw_string_into_raw_literal_keeps_trivia() {
    // Editing a property whose value itself contains quotes/spaces does not
    // disturb adjacent raw strings.
    let src = fixture("raw_string");
    let doc = parse(&src);
    let version = SourceVersion::new(src.clone());
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    plan.set_property(doc.get("note").unwrap(), "raw", KdlValue::from("x#\"y"))
        .unwrap();
    let edits = plan.edits().to_vec();
    let out = plan.apply().unwrap();

    // The default value renderer uses classic quoted strings with escapes
    // (`x#"y` -> `"x#\"y"`); the parser accepts it and the bytes outside the
    // old raw literal are untouched.
    assert!(out.contains("raw=\"x#\\\"y\""));
    assert_tree_equiv(&src, &edits, &out, |doc| {
        doc.get_mut("note").unwrap()["raw"] = KdlValue::from("x#\"y");
    });
}

// --- comments.kdl -----------------------------------------------------------

#[test]
fn comments_are_preserved_around_edit() {
    let src = fixture("comments");
    let doc = parse(&src);
    let version = SourceVersion::new(src.clone());
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    let node = doc.get("node").unwrap();
    plan.set_argument(node, 1, KdlValue::Integer(42)).unwrap();
    let child = node.children().unwrap().get("child").unwrap();
    plan.set_argument(child, 0, KdlValue::Bool(true)).unwrap();

    let edits = plan.edits().to_vec();
    let out = plan.apply().unwrap();

    assert_tree_equiv(&src, &edits, &out, |doc| {
        doc.get_mut("node")
            .unwrap()
            .entry_mut(1)
            .unwrap()
            .set_value(KdlValue::Integer(42));
        doc.get_mut("node")
            .unwrap()
            .children_mut()
            .as_mut()
            .unwrap()
            .get_mut("child")
            .unwrap()
            .entry_mut(0)
            .unwrap()
            .set_value(KdlValue::Bool(true));
    });

    // Every comment byte survives.
    assert!(out.contains("// leading comment stays"));
    assert!(out.contains("/* inline */"));
    assert!(out.contains("/-dropped kept"));
    assert!(out.contains("// trailing child comment"));
    assert!(out.contains("/- whole /- node stays"));
    assert!(out.contains("// final comment"));
}

#[test]
fn delete_node_keeps_leading_and_trailing_trivia() {
    let src = fixture("comments");
    let doc = parse(&src);
    let version = SourceVersion::new(src.clone());
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    let other = doc
        .get("node")
        .unwrap()
        .children()
        .unwrap()
        .get("other")
        .unwrap();
    plan.delete_node(other).unwrap();
    let out = plan.apply().unwrap();

    // The slash-dashed sibling text and the terminator survive; only the
    // node's own span (`other key="v"`) disappears.
    assert!(!out.contains("other key="));
    assert!(out.contains("/- whole /- node stays"));
    assert!(out.contains("child 1 // trailing child comment"));

    let mut edited = parse(&out);
    edited.clear_format_recursive();
    let mut oracle = parse(&src);
    let kids = oracle
        .get_mut("node")
        .unwrap()
        .children_mut()
        .as_mut()
        .unwrap();
    let pos = kids
        .nodes()
        .iter()
        .position(|n| n.name().value() == "other")
        .unwrap();
    kids.nodes_mut().remove(pos);
    oracle.clear_format_recursive();
    assert_eq!(edited, oracle);
}

#[test]
fn rename_node_preserves_annotations_and_children() {
    let inline = "  (t)\"old\" a=1 { // head\n    kid 2\n}\n";
    let doc = parse(inline);
    let version = SourceVersion::new(inline);
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    plan.rename_node(&doc.nodes()[0], KdlIdentifier::from("new"))
        .unwrap();
    let out = plan.apply().unwrap();

    assert_eq!(out, "  (t)new a=1 { // head\n    kid 2\n}\n");
    assert_tree_equiv(inline, plan.edits(), &out, |doc| {
        doc.nodes_mut()[0].set_name("new");
    });
}

// --- multiline_value.kdl ----------------------------------------------------

#[test]
fn multiline_values_and_later_nodes_keep_bytes() {
    let src = fixture("multiline_value");
    let doc = parse(&src);
    let version = SourceVersion::new(src.clone());
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    let config = doc.get("config").unwrap();
    plan.set_property(config, "text", KdlValue::from("one line"))
        .unwrap();
    plan.set_argument(doc.get("after").unwrap(), 0, KdlValue::Integer(2))
        .unwrap();

    let edits = plan.edits().to_vec();
    let out = plan.apply().unwrap();

    assert_tree_equiv(&src, &edits, &out, |doc| {
        doc.get_mut("config").unwrap()["text"] = KdlValue::from("one line");
        doc.get_mut("after")
            .unwrap()
            .entry_mut(0)
            .unwrap()
            .set_value(KdlValue::Integer(2));
    });

    // The multi-line raw value on `script` survives verbatim, including its
    // interior newlines and indentation.
    assert!(out.contains("script #\"\"\"\n    raw\n    multi\n    \"\"\"#\n"));
    assert!(out.contains("text=\"one line\""));
    assert!(out.ends_with("after 2\n"));
}

// --- crlf.kdl ---------------------------------------------------------------

#[test]
fn crlf_line_endings_are_byte_preserved() {
    let src = fixture("crlf");
    assert!(src.contains("\r\n"), "fixture must actually use CRLF");
    let doc = parse(&src);
    let version = SourceVersion::new(src.clone());
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    let name_node = doc.get("name").unwrap();
    plan.set_property(name_node, "prop", KdlValue::from("new"))
        .unwrap();
    plan.rename_node(doc.get("keep").unwrap(), KdlIdentifier::from("kept"))
        .unwrap();

    let edits = plan.edits().to_vec();
    let out = plan.apply().unwrap();

    assert_eq!(
        out,
        "// crlf fixture\r\nname prop=new arg=1\r\nkept (ty)\"x\"\r\n"
    );
    // No line ending was rewritten.
    assert!(!out.replace("\r\n", "").contains('\r'));
    assert!(out.contains("\r\n"));
    assert_tree_equiv(&src, &edits, &out, |doc| {
        doc.get_mut("name").unwrap()["prop"] = KdlValue::from("new");
        doc.get_mut("keep").unwrap().set_name("kept");
    });
}

#[test]
fn delete_node_with_crlf() {
    let src = fixture("crlf");
    let doc = parse(&src);
    let version = SourceVersion::new(src.clone());
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    plan.delete_node(doc.get("name").unwrap()).unwrap();
    let edits = plan.edits().to_vec();
    let out = plan.apply().unwrap();

    assert_eq!(out, "// crlf fixture\r\n\r\nkeep (ty)\"x\"\r\n");
    assert!(KdlDocument::parse(&out).is_ok());
    assert_only_ranges_changed(&src, &edits, &out);
}

// --- span anchoring regression (the most dangerous failure mode) -----------

/// A property value anchor computed from the *start* instead of measured back
/// from the entry end would overwrite the key/`=` or, with a type annotation,
/// leave the old literal dangling. This is the sharpest corruption case for
/// trivia-preserving edits and is pinned here explicitly.
#[test]
fn property_value_anchor_regression_with_type_and_key() {
    let src = "n  key  =  (t)\"old value\"  next\n";
    let doc = parse(src);
    let version = SourceVersion::new(src);
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    plan.set_property(&doc.nodes()[0], "key", KdlValue::Integer(7))
        .unwrap();
    let edit = &plan.edits()[0];
    // The edit must cover the value literal only, never the key, '=' or the
    // type annotation (the annotation is dropped along with the old value).
    assert_eq!(&src[edit.start()..edit.end()], "\"old value\"");
    let out = plan.apply().unwrap();
    assert_eq!(out, "n  key  =  (t)7  next\n");
    assert_tree_equiv(src, plan.edits(), &out, |doc| {
        doc.nodes_mut()[0]["key"] = KdlValue::Integer(7);
    });
}

#[test]
fn argument_anchor_skips_type_annotation_only() {
    let src = "n (a)x (b)y\n";
    let doc = parse(src);
    let version = SourceVersion::new(src);
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    plan.set_argument(&doc.nodes()[0], 1, KdlValue::Bool(false))
        .unwrap();
    let edit = &plan.edits()[0];
    assert_eq!(&src[edit.start()..edit.end()], "y");
    let out = plan.apply().unwrap();
    assert_eq!(out, "n (a)x (b)#false\n");
}

// --- conflict detection and safety -----------------------------------------

#[test]
fn overlapping_edits_are_rejected() {
    let src = "foo 1 2 3\nbar 4\n";
    let doc = parse(src);
    let version = SourceVersion::new(src);
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    plan.set_argument(doc.get("foo").unwrap(), 0, KdlValue::Integer(9))
        .unwrap();

    // Deleting the same node contains the argument range: must be rejected,
    // and the first edit must remain untouched.
    let err = plan.delete_node(doc.get("foo").unwrap()).unwrap_err();
    assert!(err.diagnostics.len() >= 2);
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.help.as_deref().unwrap_or("").contains("guessed order"))
    );
    assert_eq!(plan.len(), 1);

    // Same target twice (identical range) is also an overlap.
    let err = plan
        .set_argument(doc.get("foo").unwrap(), 0, KdlValue::Integer(10))
        .unwrap_err();
    assert!(
        err.diagnostics[0]
            .message
            .as_deref()
            .unwrap()
            .contains("overlaps")
    );
    assert_eq!(plan.len(), 1);
}

#[test]
fn disjoint_edits_on_same_node_compose() {
    let src = "foo 1 2 3\n";
    let doc = parse(src);
    let version = SourceVersion::new(src);
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    plan.set_argument(doc.get("foo").unwrap(), 0, KdlValue::Integer(10))
        .unwrap();
    plan.set_argument(doc.get("foo").unwrap(), 2, KdlValue::Integer(30))
        .unwrap();
    let out = plan.apply().unwrap();
    assert_eq!(out, "foo 10 2 30\n");
    // Adjacent/ordered, no overlap.
    assert_eq!(plan.edits().len(), 2);
}

#[test]
fn plans_from_different_versions_cannot_merge_or_apply() {
    let src_a = "foo 1\n";
    let src_b = "foo 2\n";
    let doc_a = parse(src_a);
    let version_a = SourceVersion::new(src_a);
    let mut plan_a = EditPlan::new(&version_a, &doc_a).unwrap();
    plan_a
        .set_argument(doc_a.get("foo").unwrap(), 0, KdlValue::Integer(9))
        .unwrap();

    // Applying to wrong source text.
    assert!(plan_a.apply_to(src_b).is_err());

    let doc_b = parse(src_b);
    let version_b = SourceVersion::new(src_b);
    let mut plan_b = EditPlan::new(&version_b, &doc_b).unwrap();
    plan_b
        .set_argument(doc_b.get("foo").unwrap(), 0, KdlValue::Integer(8))
        .unwrap();

    let err = plan_a.merge(&plan_b).unwrap_err();
    assert!(
        err.diagnostics[0]
            .message
            .as_deref()
            .unwrap()
            .contains("different source versions")
    );

    // Same content, separately parsed: merge succeeds and overlaps are caught.
    let doc_a2 = parse(src_a);
    let version_a2 = SourceVersion::new(src_a);
    let mut plan_a2 = EditPlan::new(&version_a2, &doc_a2).unwrap();
    plan_a2
        .set_argument(doc_a2.get("foo").unwrap(), 0, KdlValue::Integer(7))
        .unwrap();
    assert!(plan_a.merge(&plan_a2).is_err()); // identical ranges overlap
}

#[test]
fn mutated_document_is_rejected_as_unfaithful() {
    let src = "foo 1\n";
    let mut doc = parse(src);
    doc.get_mut("foo").unwrap().set_name("bar");
    let version = SourceVersion::new(src);
    let err = EditPlan::new(&version, &doc).unwrap_err();
    assert!(
        err.diagnostics[0]
            .message
            .as_deref()
            .unwrap()
            .contains("does not faithfully render")
    );
}

#[test]
fn missing_targets_are_diagnostic_errors_not_panics() {
    let src = "foo a=1\n";
    let doc = parse(src);
    let version = SourceVersion::new(src);
    let mut plan = EditPlan::new(&version, &doc).unwrap();

    let missing_arg = plan.set_argument(doc.get("foo").unwrap(), 5, KdlValue::Null);
    assert!(missing_arg.is_err());
    let missing_prop = plan.set_property(doc.get("foo").unwrap(), "nope", KdlValue::Null);
    assert!(missing_prop.is_err());
    // Plan stays empty after failed calls.
    assert!(plan.is_empty());

    // A node from another document cannot be edited through this plan.
    let other = parse("baz 9\n");
    assert!(plan.delete_node(other.get("baz").unwrap()).is_err());
    assert!(plan.rename_node(other.get("baz").unwrap(), "x").is_err());
}

#[test]
fn diagnostic_spans_point_at_source() {
    let src = "foo 1 2\nbar 3\n";
    let doc = parse(src);
    let version = SourceVersion::new(src);
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    plan.set_argument(doc.get("foo").unwrap(), 0, KdlValue::Integer(9))
        .unwrap();
    let err = plan.set_argument(doc.get("bar").unwrap(), 0, KdlValue::Integer(9));
    // Different nodes, disjoint ranges: this must succeed; instead trigger a
    // containment error and inspect its labels.
    assert!(err.is_ok());
    let conflict = plan.delete_node(doc.get("foo").unwrap()).unwrap_err();
    let labels: Vec<_> = conflict
        .diagnostics
        .iter()
        .filter_map(|d| d.label.clone())
        .collect();
    assert!(labels.iter().any(|l| l == "overlapping edit"));
    assert!(labels.iter().any(|l| l == "conflicts here"));
    // Labels resolve against the pinned source.
    for diag in &conflict.diagnostics {
        let span = diag.span;
        assert!(span.offset() + span.len() <= src.len());
        assert!(src.get(span.offset()..span.offset() + span.len()).is_some());
    }
}

#[test]
fn edits_are_sorted_and_non_overlapping() {
    let src = "a 1\nb 2\nc 3\n";
    let doc = parse(src);
    let version = SourceVersion::new(src);
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    plan.delete_node(doc.get("c").unwrap()).unwrap();
    plan.delete_node(doc.get("a").unwrap()).unwrap();
    plan.delete_node(doc.get("b").unwrap()).unwrap();
    let starts: Vec<usize> = plan.edits().iter().map(SourceEdit::start).collect();
    assert_eq!(
        starts,
        starts
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    );
    let out = plan.apply().unwrap();
    assert_eq!(out, "\n\n\n");
    assert_tree_equiv(src, plan.edits(), &out, |doc| {
        doc.nodes_mut().clear();
    });
}

#[test]
fn empty_plan_applies_to_identical_source() {
    let src = "foo 1\n// tail\n";
    let doc = parse(src);
    let version = SourceVersion::new(src);
    let plan = EditPlan::new(&version, &doc).unwrap();
    assert!(plan.is_empty());
    assert_eq!(plan.apply().unwrap(), src);
}

#[test]
fn node_deep_in_children_is_located_absolutely() {
    let src = "outer {\n  inner {\n    leaf old=1\n  }\n}\n";
    let doc = parse(src);
    let version = SourceVersion::new(src);
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    let leaf = doc
        .get("outer")
        .unwrap()
        .children()
        .unwrap()
        .get("inner")
        .unwrap()
        .children()
        .unwrap()
        .get("leaf")
        .unwrap();
    plan.set_property(leaf, "old", KdlValue::Integer(2))
        .unwrap();
    let edits = plan.edits().to_vec();
    let out = plan.apply().unwrap();
    assert_eq!(out, "outer {\n  inner {\n    leaf old=2\n  }\n}\n");
    assert_tree_equiv(src, &edits, &out, |doc| {
        let leaf = doc
            .get_mut("outer")
            .unwrap()
            .children_mut()
            .as_mut()
            .unwrap()
            .get_mut("inner")
            .unwrap()
            .children_mut()
            .as_mut()
            .unwrap()
            .get_mut("leaf")
            .unwrap();
        leaf["old"] = KdlValue::Integer(2);
    });
}

#[test]
fn duplicate_property_targets_last_like_tree_api() {
    let src = "foo k=1 k=2\n";
    let doc = parse(src);
    let version = SourceVersion::new(src);
    let mut plan = EditPlan::new(&version, &doc).unwrap();
    plan.set_property(doc.get("foo").unwrap(), "k", KdlValue::Integer(3))
        .unwrap();
    let edits = plan.edits().to_vec();
    let out = plan.apply().unwrap();
    assert_eq!(out, "foo k=1 k=3\n");
    assert_tree_equiv(src, &edits, &out, |doc| {
        doc.get_mut("foo").unwrap()["k"] = KdlValue::Integer(3);
    });
}
