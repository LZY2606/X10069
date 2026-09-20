//! Fixture-driven tests for the positional, trivia-preserving edit API
//! (`kdl::EditPlan` / `kdl::SourceEdit`).
//!
//! These tests are intentionally self-contained: every KDL "fixture" lives
//! inline as a string and every test exercises one behaviour, so a single case
//! can be located and run with, e.g.:
//!
//! ```sh
//! cargo test --locked --all-features --test positional_edit raw_string
//! ```
//!
//! The central invariant, enforced by [`assert_edits_match_tree`], is:
//!
//! 1. the plan is built and applied against one fixed source version;
//! 2. the produced source edits are non-overlapping;
//! 3. the applied text parses successfully;
//! 4. after canonicalisation, the parsed result is equal to what the normal
//!    tree-API mutation produces.
//!
//! Canonicalisation strips formatting (`clear_format_recursive`) on both
//! sides, so the comparison is about the *semantic* AST rather than the exact
//! bytes (the whole point of the API is that the edited text keeps the
//! original bytes everywhere the tree renderer would have rewritten them).
#![cfg(feature = "span")]

use kdl::{EditErrorKind, EditKind, EditPlan, KdlDocument, KdlNode, KdlValue};

/// Result of applying a plan: the edits in source order and the new text.
struct Applied {
    edits: Vec<kdl::SourceEdit>,
    text: String,
}

fn plan_apply(src: &str, build: impl FnOnce(&mut EditPlan, &KdlDocument)) -> Applied {
    let doc: KdlDocument = src.parse().expect("fixture must parse");
    let mut plan = EditPlan::new(src.to_string());
    build(&mut plan, &doc);
    let edits = plan.edits().to_vec();
    let text = plan.apply(src).expect("plan must apply to its own source");
    Applied { edits, text }
}

/// Apply a [`KdlValue`] change through the normal tree API in the way that
/// actually re-renders the new value: set the value and drop the entry's
/// recorded formatting (otherwise the old `value_repr` is kept).
fn tree_set_arg(nodes: &mut [KdlNode], node: usize, arg: usize, value: KdlValue) {
    let entry = nodes[node].entry_mut(arg).expect("arg must exist");
    entry.set_value(value);
    entry.clear_format();
}

fn tree_set_prop(nodes: &mut [KdlNode], node: usize, key: &str, value: KdlValue) {
    let entry = nodes[node].entry_mut(key).expect("prop must exist");
    entry.set_value(value);
    entry.clear_format();
}

fn canonical(src: &str) -> KdlDocument {
    let mut doc: KdlDocument = src.parse().expect("result must parse");
    doc.clear_format_recursive();
    doc
}

/// The shared invariant: edits are non-overlapping, the result parses, and its
/// canonical AST equals the canonical tree-API result.
fn assert_edits_match_tree(
    src: &str,
    build: impl FnOnce(&mut EditPlan, &KdlDocument),
    mutate: impl FnOnce(&mut KdlDocument),
) -> Applied {
    let applied = plan_apply(src, build);

    // The edits must already be sorted and pairwise non-overlapping.
    let mut last_end = 0usize;
    for edit in &applied.edits {
        assert!(
            edit.start() >= last_end,
            "edits overlap or are out of order: {:?}",
            applied.edits
        );
        last_end = edit.end();
    }

    // Applied text must parse successfully.
    let _: KdlDocument = applied
        .text
        .parse()
        .expect("applied edits must produce valid KDL");

    // It must equal the tree-API mutation, modulo trivia/representation.
    let mut expected: KdlDocument = src.parse().expect("fixture must parse");
    mutate(&mut expected);
    assert_eq!(
        canonical(&applied.text),
        canonical(&expected.to_string()),
        "edited output did not match the tree-API mutation\nedited: {:?}\ntree:   {:?}",
        applied.text,
        expected.to_string()
    );
    applied
}

fn node_path<'a>(doc: &'a KdlDocument, path: &[usize]) -> &'a KdlNode {
    let mut node = &doc.nodes()[path[0]];
    for &index in &path[1..] {
        node = &node.children().expect("children").nodes()[index];
    }
    node
}

// ---------------------------------------------------------------------------
// Individual operations
// ---------------------------------------------------------------------------

#[test]
fn updates_an_argument_and_preserves_surrounding_trivia() {
    let src = "foo 1 2 3\nbar 4\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_argument(&doc.nodes()[0], 1, 99).unwrap();
        },
        |doc| tree_set_arg(doc.nodes_mut(), 0, 1, 99.into()),
    );
    // Minimal, byte-accurate change: only the single digit moved.
    assert_eq!(applied.text, "foo 1 99 3\nbar 4\n");
    assert_eq!(applied.edits.len(), 1);
    assert_eq!(applied.edits[0].kind(), EditKind::Argument);
    assert_eq!(applied.edits[0].span(), (6..7).into());
    assert_eq!(applied.edits[0].replacement(), "99");
}

#[test]
fn updates_a_property_and_keeps_key_equals_and_spacing() {
    // Spaces around `key = value` are legal node-space and must be preserved.
    let src = "foo  x = 1\nnext p=2\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_property(&doc.nodes()[0], "x", "zz").unwrap();
        },
        |doc| tree_set_prop(doc.nodes_mut(), 0, "x", "zz".into()),
    );
    // Only the value token changes; the weird ` x  = ` spacing is preserved.
    assert_eq!(applied.text, "foo  x = zz\nnext p=2\n");
    assert_eq!(applied.edits[0].kind(), EditKind::Property);
}

#[test]
fn updates_the_last_duplicate_property_like_the_tree_api() {
    let src = "foo k=1 k=2\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_property(&doc.nodes()[0], "k", 9).unwrap();
        },
        |doc| tree_set_prop(doc.nodes_mut(), 0, "k", 9.into()),
    );
    assert_eq!(applied.text, "foo k=1 k=9\n");
}

#[test]
fn renames_a_node_and_keeps_its_leading_comment() {
    let src = "// header comment\nold-name 1\nnext 2\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_node_name(&doc.nodes()[0], "new-name").unwrap();
        },
        |doc| {
            doc.nodes_mut()[0].set_name("new-name");
        },
    );
    assert_eq!(applied.text, "// header comment\nnew-name 1\nnext 2\n");
    assert_eq!(applied.edits[0].kind(), EditKind::NodeName);
}

#[test]
fn renaming_a_quoted_name_quotes_only_when_required() {
    let src = "\"needs quotes\" 1\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_node_name(&doc.nodes()[0], "plain").unwrap();
        },
        |doc| {
            doc.nodes_mut()[0].set_name("plain");
        },
    );
    assert_eq!(applied.text, "plain 1\n");
}

#[test]
fn deletes_the_first_node_including_its_leading_comment_and_newline() {
    let src = "// c\na 1\nb 2\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.delete_node(&doc.nodes()[0]).unwrap();
        },
        |doc| {
            doc.nodes_mut().remove(0);
        },
    );
    assert_eq!(applied.text, "b 2\n");
    assert_eq!(applied.edits[0].kind(), EditKind::DeleteNode);
    assert!(applied.edits[0].is_deletion());
}

#[test]
fn deletes_a_later_node_without_touching_indentation_elsewhere() {
    let src = "a 1\n\n  b 2\nc 3\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.delete_node(&doc.nodes()[1]).unwrap();
        },
        |doc| {
            doc.nodes_mut().remove(1);
        },
    );
    // The separator newline after `a 1` is kept; the blank line + indent that
    // belonged to `b` travel with the deleted node.
    assert_eq!(applied.text, "a 1\nc 3\n");
    let _ = applied;
}

#[test]
fn deletes_a_nested_node_from_a_child_block() {
    let src = "parent {\n  c1 1 // hi\n  c2 2\n}\nq 9\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.delete_node(node_path(doc, &[0, 0])).unwrap();
        },
        |doc| {
            doc.nodes_mut()[0]
                .children_mut()
                .as_mut()
                .unwrap()
                .nodes_mut()
                .remove(0);
        },
    );
    // Deleting the first child removes c1 and its `// hi` terminator. The
    // newline+indent that opened the child block now sits right after `{`
    // (c2 only owned its own two-space indent), so this is the byte-minimal
    // result; the semantic invariant above confirms it equals the tree result.
    assert_eq!(applied.text, "parent {  c2 2\n}\nq 9\n");
}

#[test]
fn deleting_a_later_child_preserves_indentation_of_remaining_children() {
    // When the second child is removed it takes only its own leading spaces;
    // the newline/indent that begin the child block stay with the first child.
    let src = "parent {\n  c1 1\n  c2 2\n}\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.delete_node(node_path(doc, &[0, 1])).unwrap();
        },
        |doc| {
            doc.nodes_mut()[0]
                .children_mut()
                .as_mut()
                .unwrap()
                .nodes_mut()
                .remove(1);
        },
    );
    assert_eq!(applied.text, "parent {\n  c1 1\n}\n");
}

#[test]
fn deleting_a_node_with_children_removes_the_whole_block_and_terminator() {
    let src = "before 0\nparent {\n  child 1\n}\nafter 2\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.delete_node(&doc.nodes()[1]).unwrap();
        },
        |doc| {
            doc.nodes_mut().remove(1);
        },
    );
    assert_eq!(applied.text, "before 0\nafter 2\n");
}

#[test]
fn deleting_a_node_with_a_multiline_value_removes_all_of_its_lines() {
    let src = "before 1\nnode x=#\"\"\"\nmulti\nline\n\"\"\"#\nafter 2\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.delete_node(&doc.nodes()[1]).unwrap();
        },
        |doc| {
            doc.nodes_mut().remove(1);
        },
    );
    assert_eq!(applied.text, "before 1\nafter 2\n");
}

#[test]
fn deleting_a_node_takes_its_trailing_inline_comment() {
    // A line comment acts as the node terminator and is owned by that node, so
    // removing the node removes the comment too (matching the tree API).
    let src = "a 1\nb 2 // bye\nc 3\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.delete_node(&doc.nodes()[1]).unwrap();
        },
        |doc| {
            doc.nodes_mut().remove(1);
        },
    );
    assert_eq!(applied.text, "a 1\nc 3\n");
}

#[test]
fn edits_remain_correct_with_a_leading_byte_order_mark() {
    // The parser keeps the BOM as document leading trivia while node/entry
    // spans are already offset past it. The planner must translate both
    // correctly.
    let src = "\u{feff}a 1\nb 2\n";
    let doc: KdlDocument = src.parse().unwrap();
    let mut plan = EditPlan::new(src.to_string());
    plan.set_argument(&doc.nodes()[0], 0, 99).unwrap();
    plan.set_node_name(&doc.nodes()[1], "z").unwrap();
    // Two disjoint edits (argument of `a` and name of `b`) apply cleanly.
    assert_eq!(plan.edits().len(), 2);
    assert_eq!(plan.apply(src).unwrap(), "\u{feff}a 99\nz 2\n");

    // Deleting the first node keeps the BOM (it is document, not node, trivia).
    let doc: KdlDocument = src.parse().unwrap();
    let mut delete_first = EditPlan::new(src.to_string());
    delete_first.delete_node(&doc.nodes()[0]).unwrap();
    assert_eq!(delete_first.apply(src).unwrap(), "\u{feff}b 2\n");
}

// ---------------------------------------------------------------------------
// Trivia / literal edge cases required by the spec
// ---------------------------------------------------------------------------

#[test]
fn replaces_a_raw_string_value() {
    let src = "foo #\"raw \\ value\"#\nbar 1\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_argument(&doc.nodes()[0], 0, "new").unwrap();
        },
        |doc| tree_set_arg(doc.nodes_mut(), 0, 0, "new".into()),
    );
    // The raw literal (including backslashes that need no escaping) is fully
    // replaced; the trailing newline and next node survive verbatim.
    assert_eq!(applied.text, "foo new\nbar 1\n");
}

#[test]
fn replaces_a_single_line_raw_property_value() {
    let src = "foo k=##\"a\"b\"##\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_property(&doc.nodes()[0], "k", "z").unwrap();
        },
        |doc| tree_set_prop(doc.nodes_mut(), 0, "k", "z".into()),
    );
    assert_eq!(applied.text, "foo k=z\n");
}

#[test]
fn replaces_a_multiline_raw_string_argument() {
    let src = "foo #\"\"\"\nraw \" line\nsecond\n\"\"\"#\nnext 1\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_argument(&doc.nodes()[0], 0, 42).unwrap();
        },
        |doc| tree_set_arg(doc.nodes_mut(), 0, 0, 42.into()),
    );
    // All four lines of the raw literal collapse to one minimal token.
    assert_eq!(applied.text, "foo 42\nnext 1\n");
    assert_eq!(applied.edits[0].replacement(), "42");
}

#[test]
fn replaces_a_multiline_quoted_property_value() {
    let src = "foo k=\"\"\"\nline one\nline two\n\"\"\"\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_property(&doc.nodes()[0], "k", KdlValue::Bool(true))
                .unwrap();
        },
        |doc| tree_set_prop(doc.nodes_mut(), 0, "k", true.into()),
    );
    assert_eq!(applied.text, "foo k=#true\n");
}

#[test]
fn preserves_an_existing_type_annotation_when_changing_a_value() {
    let src = "foo (uuid)1 key=(flag)#true\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_argument(&doc.nodes()[0], 0, 5).unwrap();
            plan.set_property(&doc.nodes()[0], "key", false).unwrap();
        },
        |doc| {
            tree_set_arg(doc.nodes_mut(), 0, 0, 5.into());
            tree_set_prop(doc.nodes_mut(), 0, "key", false.into());
        },
    );
    assert_eq!(applied.text, "foo (uuid)5 key=(flag)#false\n");
}

#[test]
fn works_across_crlf_line_endings() {
    let src = "a 1\r\nb 2\r\nc 3\r\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_argument(&doc.nodes()[1], 0, 7).unwrap();
            plan.set_node_name(&doc.nodes()[2], "renamed").unwrap();
        },
        |doc| {
            let entry = doc.nodes_mut()[1].entry_mut(0).unwrap();
            entry.set_value(7);
            entry.clear_format();
            doc.nodes_mut()[2].set_name("renamed");
        },
    );
    // CRLF bytes are preserved exactly.
    assert_eq!(applied.text, "a 1\r\nb 7\r\nrenamed 3\r\n");
}

#[test]
fn deleting_a_node_keeps_crlf_separators_for_other_nodes() {
    let src = "a 1\r\nb 2\r\nc 3\r\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.delete_node(&doc.nodes()[1]).unwrap();
        },
        |doc| {
            doc.nodes_mut().remove(1);
        },
    );
    assert_eq!(applied.text, "a 1\r\nc 3\r\n");
}

#[test]
fn preserves_inline_and_full_line_comments_around_unrelated_nodes() {
    let src = "// top\na 1 // keep me\r\n/* block */ b 2 {\n  // inner\n  c 3\n}\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_argument(&doc.nodes()[0], 0, 10).unwrap();
        },
        |doc| tree_set_arg(doc.nodes_mut(), 0, 0, 10.into()),
    );
    assert_eq!(
        applied.text,
        "// top\na 10 // keep me\r\n/* block */ b 2 {\n  // inner\n  c 3\n}\n"
    );
}

// ---------------------------------------------------------------------------
// Multiple, independent edits in one plan
// ---------------------------------------------------------------------------

#[test]
fn applies_several_disjoint_edits_in_source_order() {
    let src = "a 1 {\n  b x=2 y=3\n}\nc 4 // tail\n";
    let applied = assert_edits_match_tree(
        src,
        |plan, doc| {
            plan.set_node_name(&doc.nodes()[0], "A").unwrap();
            let inner = node_path(doc, &[0, 0]);
            plan.set_property(inner, "y", 30).unwrap();
            plan.delete_node(&doc.nodes()[1]).unwrap();
        },
        |doc| {
            doc.nodes_mut()[0].set_name("A");
            let entry = doc.nodes_mut()[0]
                .children_mut()
                .as_mut()
                .unwrap()
                .nodes_mut()[0]
                .entry_mut("y")
                .unwrap();
            entry.set_value(30);
            entry.clear_format();
            doc.nodes_mut().remove(1);
        },
    );
    // Edits are returned sorted by offset even though they were added out of
    // order (delete `c` was requested before the inner property is located in
    // the source... actually inner property precedes `c`, and sorting handles
    // any request order).
    let offsets: Vec<usize> = applied.edits.iter().map(|e| e.start()).collect();
    let mut sorted = offsets.clone();
    sorted.sort_unstable();
    assert_eq!(offsets, sorted);
    assert_eq!(applied.text, "A 1 {\n  b x=2 y=30\n}\n");
}

#[test]
fn two_touching_but_disjoint_edits_are_allowed() {
    // Argument spans "[2..3]" and "[4..5]" are adjacent, not overlapping.
    let src = "a 1 2\n";
    let doc: KdlDocument = src.parse().unwrap();
    let mut plan = EditPlan::new(src.to_string());
    plan.set_argument(&doc.nodes()[0], 0, 10).unwrap();
    plan.set_argument(&doc.nodes()[0], 1, 20).unwrap();
    assert_eq!(plan.edits().len(), 2);
    assert_eq!(plan.apply(src).unwrap(), "a 10 20\n");
}

// ---------------------------------------------------------------------------
// Conflict detection and same-source binding
// ---------------------------------------------------------------------------

#[test]
fn rejects_overlapping_edits_to_the_same_entry() {
    let src = "a x=1 y=2\n";
    let doc: KdlDocument = src.parse().unwrap();
    let mut plan = EditPlan::new(src.to_string());
    plan.set_property(&doc.nodes()[0], "x", 1).unwrap();
    let err = plan
        .set_property(&doc.nodes()[0], "x", 2)
        .expect_err("rewriting the same value twice must be rejected");
    assert_eq!(err.kind, EditErrorKind::OverlappingEdits);
    assert!(err.other_span.is_some());
}

#[test]
fn rejects_an_edit_inside_a_node_that_is_being_deleted() {
    let src = "parent {\n  child 1\n}\nnext 2\n";
    let doc: KdlDocument = src.parse().unwrap();
    let child = doc.nodes()[0].children().unwrap().nodes()[0].clone();
    let mut plan = EditPlan::new(src.to_string());
    plan.delete_node(&doc.nodes()[0]).unwrap();
    // The argument span is strictly contained within the deleted node range.
    let err = plan
        .set_argument(&child, 0, 9)
        .expect_err("containment must be rejected, not reordered");
    assert_eq!(err.kind, EditErrorKind::OverlappingEdits);
    assert!(err.message().contains("overlap"));
}

#[test]
fn rejects_crossing_edits_spanning_a_deletion_boundary() {
    // Deleting node `b` covers its leading trivia + body + terminator; a
    // rename of `b` itself sits inside that range and so crosses/contains it.
    let src = "a 1\nb 2\nc 3\n";
    let doc: KdlDocument = src.parse().unwrap();
    let target = doc.nodes()[1].clone();
    let mut plan = EditPlan::new(src.to_string());
    plan.delete_node(&target).unwrap();
    let err = plan.set_node_name(&target, "z").unwrap_err();
    assert_eq!(err.kind, EditErrorKind::OverlappingEdits);
}

#[test]
fn rejects_an_edit_against_a_different_source_version() {
    let src = "a 1\nb 2\n";
    // Same length, different content -> the value anchor byte check fails even
    // though offsets are identical.
    let other = "a 7\nb 2\n";
    let foreign: KdlDocument = other.parse().unwrap();
    let mut plan = EditPlan::new(src.to_string());
    let err = plan
        .set_argument(&foreign.nodes()[0], 0, 1)
        .expect_err("a node from another source must not be accepted");
    assert_eq!(err.kind, EditErrorKind::MismatchedSource);
}

#[test]
fn rejects_applying_to_a_changed_source_even_of_the_same_length() {
    let src = "a 1\nb 2\n";
    let doc: KdlDocument = src.parse().unwrap();
    let mut plan = EditPlan::new(src.to_string());
    plan.set_argument(&doc.nodes()[0], 0, 9).unwrap();
    let changed = "a 5\nb 2\n";
    assert_eq!(changed.len(), src.len(), "fixture keeps length equal");
    let err = plan
        .apply(changed)
        .expect_err("stale source must be rejected");
    assert_eq!(err.kind, EditErrorKind::StaleSource);
}

#[test]
fn rejects_applying_when_offsets_would_shift() {
    let src = "a 1\nb 2\n";
    let doc: KdlDocument = src.parse().unwrap();
    let mut plan = EditPlan::new(src.to_string());
    plan.set_argument(&doc.nodes()[1], 0, 9).unwrap();
    let changed = "a 1 extra\nb 2\n";
    let err = plan
        .apply(changed)
        .expect_err("different-length source rejected");
    assert_eq!(err.kind, EditErrorKind::StaleSource);
}

#[test]
fn reports_not_found_for_missing_argument_and_property() {
    let src = "a 1 k=2\n";
    let doc: KdlDocument = src.parse().unwrap();
    let mut plan = EditPlan::new(src.to_string());
    let arg_err = plan.set_argument(&doc.nodes()[0], 5, 1).unwrap_err();
    assert_eq!(arg_err.kind, EditErrorKind::NotFound);
    let prop_err = plan.set_property(&doc.nodes()[0], "nope", 1).unwrap_err();
    assert_eq!(prop_err.kind, EditErrorKind::NotFound);
}

#[test]
fn rejects_an_invalid_replacement_name() {
    use kdl::KdlIdentifier;
    let src = "a 1\n";
    let doc: KdlDocument = src.parse().unwrap();
    let mut plan = EditPlan::new(src.to_string());
    // A well-behaved value with an explicitly broken source representation:
    // the rendered text contains a stray quote and cannot parse as an
    // identifier, so the planner refuses rather than emit corrupt KDL.
    let mut bad = KdlIdentifier::from("ok");
    bad.set_repr("o\"k".to_string());
    let err = plan
        .set_node_name(&doc.nodes()[0], bad)
        .expect_err("an unparsable replacement must be rejected");
    assert_eq!(err.kind, EditErrorKind::InvalidReplacement);
    assert!(err.help().is_some());
}

#[test]
fn diagnostics_carry_source_spans_and_labels() {
    // The diagnostic machinery (miette) must be able to read the labelled span.
    use miette::Diagnostic;
    let src = "a x=1\n";
    let doc: KdlDocument = src.parse().unwrap();
    let mut plan = EditPlan::new(src.to_string());
    plan.set_property(&doc.nodes()[0], "x", 1).unwrap();
    let err = plan.set_property(&doc.nodes()[0], "x", 2).unwrap_err();
    let labels: Vec<_> = err.labels().unwrap().collect();
    assert_eq!(labels.len(), 2);
    let code = err
        .source_code()
        .unwrap()
        .read_span(labels[0].inner(), 0, 0)
        .unwrap();
    assert_eq!(std::str::from_utf8(code.data()).unwrap(), "1");
}
