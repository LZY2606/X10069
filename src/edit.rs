//! Located, trivia-preserving editing of parsed KDL documents.
//!
//! The regular KDL tree API (see [`crate::KdlDocument`], [`crate::KdlNode`],
//! [`crate::KdlEntry`]) preserves formatting on a full round-trip, but any
//! mutation forces a re-render of the whole subtree it touched. For a caller
//! that wants the smallest possible change to an existing document — keeping
//! untouched whitespace, newlines, comments and the original literal spelling
//! of every value it did not touch — that is both expensive and surprising.
//!
//! This module provides a "source edit" workflow instead:
//!
//! 1. Parse a document and build a [`SourceVersion`] from the exact bytes that
//!    were parsed. The version pins the edit plan to one source snapshot.
//! 2. Create an [`EditPlan`] for that version and register located edits
//!    against nodes and entries the parser already carries byte spans for:
//!    [`EditPlan::rename_node`], [`EditPlan::set_argument`],
//!    [`EditPlan::set_property`] and [`EditPlan::delete_node`].
//! 3. Ask for the plan's non-overlapping [`SourceEdit`]s with
//!    [`EditPlan::edits`], or apply them directly with [`EditPlan::apply`].
//!
//! Every byte outside the ranges covered by returned edits is left exactly as
//! it was, including CRLF line endings, raw strings, multi-line strings and
//! slash-dashed comments. Overlapping and containment edits are rejected with
//! a [`KdlError`] instead of being ordered by guesswork, and plans that were
//! built for a different source version cannot be merged or applied.
//!
//! All ranges are absolute byte offsets into the source the
//! [`SourceVersion`] was created from. They are always on UTF-8 boundaries
//! because the parser tracks positions in bytes.
//!
//! ```rust
//! use kdl::{KdlDocument, KdlValue};
//! use kdl::edit::{EditPlan, SourceVersion};
//!
//! let src = "name prop=\"old\" // keep me\n";
//! let doc = KdlDocument::parse(src).unwrap();
//! let version = SourceVersion::new(src);
//! let mut plan = EditPlan::new(&version, &doc).unwrap();
//! plan.set_property(&doc.nodes()[0], "prop", KdlValue::from("new"))
//!     .unwrap();
//!
//! let out = plan.apply().unwrap();
//! assert_eq!(out, "name prop=new // keep me\n");
//! ```

use std::sync::Arc;

use miette::{Severity, SourceSpan};

use crate::{KdlDiagnostic, KdlDocument, KdlEntry, KdlError, KdlIdentifier, KdlNode, KdlValue};

/// A pinned snapshot of the source bytes a set of edits is built against.
///
/// A [`SourceVersion`] is created from the exact `&str` that was handed to
/// [`KdlDocument::parse`]. It keeps that source alive (so returned
/// [`SourceEdit`] offsets always resolve) and fingerprints it, which lets an
/// [`EditPlan`] reject edits that were planned for a different snapshot
/// instead of silently splicing into the wrong bytes.
///
/// Two versions are considered the same precisely when their source text is
/// byte-for-byte identical.
#[derive(Debug, Clone)]
pub struct SourceVersion {
    input: Arc<String>,
    fingerprint: u64,
}

impl SourceVersion {
    /// Pins a source string as an edit version.
    ///
    /// `input` should be the exact text passed to [`KdlDocument::parse`] for
    /// the document edits are planned against.
    pub fn new(input: impl Into<String>) -> Self {
        let input = Arc::new(input.into());
        let fingerprint = fnv1a64(input.as_bytes());
        Self { input, fingerprint }
    }

    /// The original source bytes this version was created from.
    pub fn source(&self) -> &str {
        self.input.as_str()
    }

    /// Length of the pinned source in bytes.
    pub fn len(&self) -> usize {
        self.input.len()
    }

    /// Returns true if the pinned source is empty.
    pub fn is_empty(&self) -> bool {
        self.input.is_empty()
    }

    fn same_version(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint && self.input == other.input
    }
}

impl PartialEq for SourceVersion {
    fn eq(&self, other: &Self) -> bool {
        self.same_version(other)
    }
}

impl Eq for SourceVersion {}

/// A single, located replacement against a [`SourceVersion`]'s source bytes.
///
/// Edits are half-open byte ranges (`span..span + span.len()`); applying one
/// replaces exactly that range with [`SourceEdit::replacement`] and leaves
/// every other byte untouched. A deletion is represented by an empty
/// replacement.
///
/// Edits returned from an [`EditPlan`] are guaranteed not to overlap, so they
/// can be applied independently or in source order without range fixups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceEdit {
    /// The byte range in the pinned source that is replaced by this edit.
    pub span: SourceSpan,
    /// The text to substitute in place of the bytes covered by `span`.
    pub replacement: String,
    kind: EditKind,
}

impl SourceEdit {
    /// What kind of located edit produced this source edit.
    pub fn kind(&self) -> EditKind {
        self.kind
    }

    /// Start byte offset of this edit in the pinned source.
    pub fn start(&self) -> usize {
        self.span.offset()
    }

    /// End byte offset (exclusive) of this edit in the pinned source.
    pub fn end(&self) -> usize {
        self.span.offset() + self.span.len()
    }
}

/// The kind of semantic change a [`SourceEdit`] performs.
///
/// This is metadata describing a planned edit; the actual text work is always
/// the byte replacement carried by [`SourceEdit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditKind {
    /// A node name was replaced.
    RenameNode,
    /// A positional argument's value was replaced.
    SetArgument,
    /// A property's value was replaced (the key is untouched).
    SetProperty,
    /// A whole node was removed.
    DeleteNode,
}

/// A plan of located edits against one parsed document and one pinned
/// [`SourceVersion`].
///
/// Register edits with the `set_*`/`rename_*`/`delete_*` methods. Each call
/// resolves its target through the spans the parser attached to the AST, so
/// the caller can navigate with the regular tree API (for example
/// [`KdlDocument::get`] or `children()`) and still end up with byte-level
/// edits.
///
/// Edits are rejected rather than guessed at when:
///
/// * the target node does not belong to the plan's document,
/// * a target argument/property does not exist,
/// * a new edit's range overlaps (including containment of) an already
///   planned edit, or is exactly the same range,
/// * a range would run past the source or land inside a UTF-8 boundary, or
/// * a merged plan was made for a different [`SourceVersion`].
///
/// Errors are reported through the existing [`KdlError`] model, with labeled
/// spans pointing at the offending source locations.
#[derive(Debug)]
pub struct EditPlan<'doc> {
    version: SourceVersion,
    doc: &'doc KdlDocument,
    edits: Vec<SourceEdit>,
}

impl<'doc> EditPlan<'doc> {
    /// Creates an empty plan for `doc`, pinned to `version`.
    ///
    /// Returns an error if `doc` does not round-trip to the exact bytes of
    /// `version` (for example if it was mutated after parsing, or if it was
    /// parsed from different text): locating edits against a document whose
    /// spans are not faithful to the source would be unsafe.
    pub fn new(version: &SourceVersion, doc: &'doc KdlDocument) -> Result<Self, KdlError> {
        if doc.to_string() != version.source() {
            return Err(KdlError {
                input: version.input.clone(),
                diagnostics: vec![diagnostic(
                    version,
                    (0..version.len()).into(),
                    "Document does not faithfully render the pinned source".into(),
                    Some("document".into()),
                    Some(
                        "Edit plans require an unmodified parse of exactly the \
                         pinned source text."
                            .into(),
                    ),
                )],
            });
        }
        Ok(Self {
            version: version.clone(),
            doc,
            edits: Vec::new(),
        })
    }

    /// The [`SourceVersion`] this plan is pinned to.
    pub fn version(&self) -> &SourceVersion {
        &self.version
    }

    /// Number of edits registered on this plan so far.
    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// Returns true if no edits have been registered.
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Returns the planned edits in ascending source order.
    ///
    /// The edits never overlap and can be applied directly in this order (or
    /// reverse order) without adjusting offsets.
    pub fn edits(&self) -> &[SourceEdit] {
        &self.edits
    }

    /// Replaces `node`'s name with `name`, keeping its type annotation,
    /// entries, children and all surrounding trivia untouched.
    pub fn rename_node(
        &mut self,
        node: &KdlNode,
        name: impl Into<KdlIdentifier>,
    ) -> Result<(), KdlError> {
        self.require_owned(node)?;
        let new_name: KdlIdentifier = name.into();
        let rendered = new_name.to_string();
        let span = node.name().span();
        let expected = self.slice(span)?;
        if expected != node.name().to_string() {
            return Err(self.stale(span, "node name", expected, &node.name().to_string()));
        }
        self.push(span, rendered, EditKind::RenameNode)
    }

    /// Replaces the value of positional argument `index` on `node`.
    ///
    /// Only the value literal (including a type annotation if present) is
    /// replaced; whitespace, slash-dash comments, the argument's position and
    /// the node's other entries are preserved.
    pub fn set_argument(
        &mut self,
        node: &KdlNode,
        index: usize,
        value: impl Into<KdlValue>,
    ) -> Result<(), KdlError> {
        self.require_owned(node)?;
        let entry = match argument(node, index) {
            Some(entry) => entry,
            None => {
                return Err(self.target_error(
                    node,
                    format!("node has no argument at index {index}"),
                    format!("only {} argument(s) present", argument_count(node)),
                ));
            }
        };
        let span = self.literal_span(entry)?;
        self.push(span, value.into().to_string(), EditKind::SetArgument)
    }

    /// Replaces the value of property `key` on `node`.
    ///
    /// The property key, `=` spacing, type annotation and all trivia outside
    /// the value literal are preserved. As with [`KdlNode::get`], the *last*
    /// property with a duplicate name is targeted.
    pub fn set_property(
        &mut self,
        node: &KdlNode,
        key: &str,
        value: impl Into<KdlValue>,
    ) -> Result<(), KdlError> {
        self.require_owned(node)?;
        let entry = match node.entry(key) {
            Some(entry) => entry,
            None => {
                return Err(self.target_error(
                    node,
                    format!("node has no property named {key:?}"),
                    "add the property with the tree API before planning a located edit".into(),
                ));
            }
        };
        let span = self.literal_span(entry)?;
        self.push(span, value.into().to_string(), EditKind::SetProperty)
    }

    /// Deletes `node` entirely, replacing the node's own span (from its type
    /// annotation/name through its children block and pre-terminator trivia)
    /// together with a trailing `;` or single-line-comment terminator when
    /// there is one.
    ///
    /// A plain newline terminator is *not* removed: keeping it preserves the
    /// surrounding blank lines and indentation, and still produces valid
    /// KDL. Leading whitespace/comments and all trailing trivia remain in
    /// place byte-for-byte. When the terminator is a `;` or a single-line
    /// comment, leaving it behind would be a syntax error or a dangling
    /// comment attached to the previous node, so it is included in the
    /// deletion instead.
    pub fn delete_node(&mut self, node: &KdlNode) -> Result<(), KdlError> {
        self.require_owned(node)?;
        let span = self.delete_span(node)?;
        self.push(span, String::new(), EditKind::DeleteNode)
    }

    /// Merges `other` into this plan.
    ///
    /// Both plans must target the same [`SourceVersion`]. The combined edit
    /// set must remain non-overlapping; otherwise an error naming both
    /// conflicting ranges is returned and `self` is left untouched.
    pub fn merge(&mut self, other: &EditPlan<'_>) -> Result<(), KdlError> {
        if !self.version.same_version(&other.version) {
            return Err(KdlError {
                input: self.version.input.clone(),
                diagnostics: vec![diagnostic(
                    &self.version,
                    (0..self.version.len()).into(),
                    "Cannot merge edit plans built from different source versions".into(),
                    Some("source version mismatch".into()),
                    Some(
                        "Rebuild both plans from the same parsed source before merging edits."
                            .into(),
                    ),
                )],
            });
        }
        let mut combined = self.edits.clone();
        for edit in &other.edits {
            self.check_overlap(&combined, edit)?;
            combined.push(edit.clone());
        }
        combined.sort_by_key(|e| e.start());
        self.edits = combined;
        Ok(())
    }

    /// Applies the planned edits to the pinned source and returns the new
    /// source text.
    ///
    /// The returned string is the input bytes with each edit's range replaced.
    pub fn apply(&self) -> Result<String, KdlError> {
        self.apply_to(self.version.source())
    }

    /// Applies the planned edits to `source`, after verifying that `source` is
    /// the exact text this plan was pinned to.
    pub fn apply_to(&self, source: &str) -> Result<String, KdlError> {
        if source != self.version.source() {
            return Err(KdlError {
                input: self.version.input.clone(),
                diagnostics: vec![diagnostic(
                    &self.version,
                    (0..self.version.len()).into(),
                    "Refusing to apply edits planned against a different source".into(),
                    Some("source version mismatch".into()),
                    Some("Recreate the edit plan for this source before applying it.".into()),
                )],
            });
        }
        let mut out = String::with_capacity(source.len());
        let mut cursor = 0usize;
        for edit in &self.edits {
            out.push_str(&source[cursor..edit.start()]);
            out.push_str(&edit.replacement);
            cursor = edit.end();
        }
        out.push_str(&source[cursor..]);
        Ok(out)
    }

    // --- private helpers -------------------------------------------------

    /// Computes the byte range removed when deleting a node. The parser's
    /// node span stops just before the terminator; a `;` or single-line
    /// comment terminator must be swallowed as well to keep the result valid
    /// and to avoid re-homing a comment onto an adjacent node.
    fn delete_span(&self, node: &KdlNode) -> Result<SourceSpan, KdlError> {
        let span = node.span();
        let mut end = span.offset() + span.len();
        if let Some(fmt) = node.format()
            && (fmt.terminator.starts_with(';') || fmt.terminator.starts_with("//"))
        {
            end += fmt.before_terminator.len();
            end += fmt.terminator.len();
        }
        let resolved: SourceSpan = (span.offset()..end).into();
        self.slice(resolved)?;
        Ok(resolved)
    }

    fn require_owned(&self, node: &KdlNode) -> Result<(), KdlError> {
        if owns_node(self.doc, node) {
            Ok(())
        } else {
            Err(KdlError {
                input: self.version.input.clone(),
                diagnostics: vec![diagnostic(
                    &self.version,
                    node.span(),
                    "Node does not belong to this edit plan's document".into(),
                    Some("foreign node".into()),
                    Some(
                        "Locate the node through the document this plan was \
                         built from (walk its nodes() and children())."
                            .into(),
                    ),
                )],
            })
        }
    }

    fn slice(&self, span: SourceSpan) -> Result<&str, KdlError> {
        let range = span.offset()..span.offset() + span.len();
        match self.version.source().get(range.clone()) {
            Some(s) => Ok(s),
            None => Err(KdlError {
                input: self.version.input.clone(),
                diagnostics: vec![diagnostic(
                    &self.version,
                    clamped_span(&self.version, span),
                    format!(
                        "Edit range {}..{} is outside the {}-byte source or splits a UTF-8 boundary",
                        range.start,
                        range.end,
                        self.version.len()
                    ),
                    Some("out of bounds".into()),
                    None,
                )],
            }),
        }
    }

    fn stale(&self, span: SourceSpan, what: &str, found: &str, expected: &str) -> KdlError {
        KdlError {
            input: self.version.input.clone(),
            diagnostics: vec![diagnostic(
                &self.version,
                span,
                format!("Source text at {what} no longer matches the parsed AST"),
                Some("stale edit".into()),
                Some(format!(
                    "source has {found:?}, but the parsed AST rendered {expected:?}; reparse before editing"
                )),
            )],
        }
    }

    fn target_error(&self, node: &KdlNode, message: String, help: String) -> KdlError {
        KdlError {
            input: self.version.input.clone(),
            diagnostics: vec![diagnostic(
                &self.version,
                node.span(),
                message,
                Some("edit target".into()),
                Some(help),
            )],
        }
    }

    /// Resolves the exact source range of an entry's value literal, i.e. the
    /// range whose replacement changes the value while preserving the
    /// property key, type annotation and every surrounding byte.
    ///
    /// The entry's span covers `name=value` for properties and `value` for
    /// arguments, where both `name` and `value` are anchored at the start and
    /// end of that span respectively (leading whitespace is stored in the
    /// entry's `leading` trivia, outside the span). Counting the key, equals
    /// sign and type-annotation bytes *forward* from the span start is
    /// therefore independent of how the literal itself is spelled (quoted,
    /// raw, multi-line, hexadecimal, ...).
    fn literal_span(&self, entry: &KdlEntry) -> Result<SourceSpan, KdlError> {
        let span = entry.span();
        let end = span.offset() + span.len();
        let mut start = span.offset();
        if let Some(name) = entry.name() {
            start += name.to_string().len();
            if let Some(fmt) = entry.format() {
                start += fmt.after_key.len();
                start += 1; // '='
                start += fmt.after_eq.len();
            } else {
                start += 1; // '='
            }
        }
        if entry.ty().is_some() {
            start += 1; // '('
            if let Some(fmt) = entry.format() {
                start += fmt.before_ty_name.len();
            }
            start += entry.ty().map_or(0, |ty| ty.to_string().len());
            if let Some(fmt) = entry.format() {
                start += fmt.after_ty_name.len();
            }
            start += 1; // ')'
            if let Some(fmt) = entry.format() {
                start += fmt.after_ty.len();
            }
        }
        if start > end {
            return Err(KdlError {
                input: self.version.input.clone(),
                diagnostics: vec![diagnostic(
                    &self.version,
                    span,
                    "Could not locate the entry's value literal inside its span".into(),
                    Some("malformed entry span".into()),
                    Some("This is an internal mismatch between parsed spans and trivia.".into()),
                )],
            });
        }
        let resolved: SourceSpan = (start..end).into();
        // Validate bounds and that the range is on a UTF-8 boundary.
        self.slice(resolved)?;
        Ok(resolved)
    }

    fn push(
        &mut self,
        span: SourceSpan,
        replacement: String,
        kind: EditKind,
    ) -> Result<(), KdlError> {
        // Validate range against source (bounds + UTF-8 boundaries).
        self.slice(span)?;
        let edit = SourceEdit {
            span,
            replacement,
            kind,
        };
        self.check_overlap(&self.edits, &edit)?;
        self.edits.push(edit);
        self.edits.sort_by_key(|e| e.start());
        Ok(())
    }

    fn check_overlap(&self, existing: &[SourceEdit], edit: &SourceEdit) -> Result<(), KdlError> {
        for prior in existing {
            if edit.start() < prior.end() && prior.start() < edit.end() {
                return Err(KdlError {
                    input: self.version.input.clone(),
                    diagnostics: vec![
                        diagnostic(
                            &self.version,
                            edit.span,
                            "Edit overlaps with an already planned edit".into(),
                            Some("overlapping edit".into()),
                            Some(
                                "Conflicting edits are rejected instead of being \
                                 applied in a guessed order. Combine them into a \
                                 single edit on the larger node."
                                    .into(),
                            ),
                        ),
                        diagnostic(
                            &self.version,
                            prior.span,
                            "Previously planned edit".into(),
                            Some("conflicts here".into()),
                            None,
                        ),
                    ],
                });
            }
        }
        Ok(())
    }
}

fn argument(node: &KdlNode, index: usize) -> Option<&KdlEntry> {
    node.entries()
        .iter()
        .filter(|e| e.name().is_none())
        .nth(index)
}

fn argument_count(node: &KdlNode) -> usize {
    node.entries().iter().filter(|e| e.name().is_none()).count()
}

fn owns_node(doc: &KdlDocument, needle: &KdlNode) -> bool {
    fn walk(children: &KdlDocument, needle: &KdlNode) -> bool {
        children.nodes().iter().any(|node| {
            std::ptr::addr_eq(node, needle)
                || node
                    .children()
                    .is_some_and(|child_doc| walk(child_doc, needle))
        })
    }
    walk(doc, needle)
}

fn diagnostic(
    version: &SourceVersion,
    span: SourceSpan,
    message: String,
    label: Option<String>,
    help: Option<String>,
) -> KdlDiagnostic {
    KdlDiagnostic {
        input: version.input.clone(),
        span,
        message: Some(message),
        label,
        help,
        severity: Severity::Error,
    }
}

fn clamped_span(version: &SourceVersion, span: SourceSpan) -> SourceSpan {
    let start = span.offset().min(version.len());
    let end = (span.offset() + span.len()).min(version.len());
    (start..end.max(start)).into()
}

/// FNV-1a 64-bit fingerprint, used only to cheaply short-circuit source
/// comparisons; equality is still confirmed with a full string compare.
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x00000100000001b3;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}
