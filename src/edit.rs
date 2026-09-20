//! Positional, trivia-preserving source edits.
//!
//! Parsing a [`KdlDocument`](crate::KdlDocument) keeps every byte of the
//! original input: whitespace, newlines, comments and the exact textual
//! representation of every value. Mutating the tree and re-rendering the whole
//! document ([`KdlDocument::to_string`](crate::KdlDocument::to_string))
//! therefore makes it hard to predict how unrelated parts of the original text
//! are affected.
//!
//! The types in this module let a caller describe *positional* edits instead.
//! An [`EditPlan`] is created from the exact source string that was parsed and
//! turns high level operations on a node (renaming it, changing an argument
//! or property, or deleting it) into a set of non-overlapping byte
//! [`SourceEdit`]s. [`EditPlan::apply`] splices those edits into a copy of the
//! source, leaving every byte that belongs to untouched trivia completely
//! alone.
//!
//! Every operation is verified against the source it belongs to:
//!
//! * the span of the node or entry being edited must still point at the exact
//!   original bytes, so a node from a different parse (or a different source
//!   version) is rejected with an [`EditError`] instead of silently editing
//!   the wrong place;
//! * the plan refuses overlapping or containing edits rather than guessing
//!   which order to apply them in;
//! * replacement values are rendered with the normal tree renderer and
//!   re-parsed, guaranteeing the result is valid KDL.
//!
//! These types are only available when the default `span` feature is enabled.
//!
//! # Example
//!
//! ```
//! use kdl::{EditPlan, KdlDocument};
//!
//! let src = "// keep this comment\nserver host=\"localhost\" port=8000\n";
//! let doc: KdlDocument = src.parse().unwrap();
//!
//! let mut plan = EditPlan::new(src.to_string());
//! let server = doc.get("server").unwrap();
//! plan.set_property(server, "port", 9000).unwrap();
//!
//! // Only `8000` is rewritten; the comment, spacing and the other property
//! // keep their exact original bytes.
//! assert_eq!(
//!     plan.apply(src).unwrap(),
//!     "// keep this comment\nserver host=\"localhost\" port=9000\n"
//! );
//! ```
//!
//! Overlapping edits are rejected rather than applied in a guessed order:
//!
//! ```
//! use kdl::{EditErrorKind, EditPlan, KdlDocument};
//!
//! let src = "parent {\n  child 1\n}\n";
//! let doc: KdlDocument = src.parse().unwrap();
//! let child = doc.get("parent").unwrap().children().unwrap().get("child").unwrap().clone();
//!
//! let mut plan = EditPlan::new(src.to_string());
//! plan.delete_node(&doc.nodes()[0]).unwrap();
//! // Editing something *inside* the node being deleted is a containment.
//! let err = plan.set_argument(&child, 0, 2).unwrap_err();
//! assert_eq!(err.kind, EditErrorKind::OverlappingEdits);
//! ```

use std::{error::Error, fmt, fmt::Display, sync::Arc};

use miette::{Diagnostic, LabeledSpan, Severity, SourceSpan};

use crate::{KdlEntry, KdlIdentifier, KdlNode, KdlValue, v2_parser};

/// A single, byte-accurate replacement inside a source string.
///
/// A `SourceEdit` replaces the bytes in `span` with `replacement`. An empty
/// `replacement` is a deletion. The spans produced by [`EditPlan`] are
/// guaranteed not to overlap.
///
/// Spans are byte offsets into the source string the plan was created from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceEdit {
    /// What kind of high level operation produced this edit.
    pub kind: EditKind,
    /// Byte span in the original source that is replaced.
    pub span: SourceSpan,
    /// Text that is substituted for `span`.
    pub replacement: String,
}

impl SourceEdit {
    /// The kind of operation this edit performs.
    pub fn kind(&self) -> EditKind {
        self.kind
    }

    /// The byte span replaced by this edit.
    pub fn span(&self) -> SourceSpan {
        self.span
    }

    /// Start byte offset (inclusive) of this edit.
    pub fn start(&self) -> usize {
        self.span.offset()
    }

    /// End byte offset (exclusive) of this edit.
    pub fn end(&self) -> usize {
        self.span.offset() + self.span.len()
    }

    /// The text substituted for the original span.
    pub fn replacement(&self) -> &str {
        &self.replacement
    }

    /// `true` if the edit removes bytes without inserting any.
    pub fn is_deletion(&self) -> bool {
        self.replacement.is_empty()
    }
}

/// The high level operation a [`SourceEdit`] was created from.
///
/// This is primarily useful for diagnostics and for tooling that wants to
/// report what a plan is about to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EditKind {
    /// A node name is being changed.
    NodeName,
    /// A positional argument is being changed.
    Argument,
    /// A property value is being changed.
    Property,
    /// A node (and its own trivia/terminator) is being removed.
    DeleteNode,
}

impl Display for EditKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditKind::NodeName => write!(f, "update node name"),
            EditKind::Argument => write!(f, "update argument"),
            EditKind::Property => write!(f, "update property"),
            EditKind::DeleteNode => write!(f, "delete node"),
        }
    }
}

/// The category of an [`EditError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EditErrorKind {
    /// A span fell outside the source or did not line up with its node.
    MismatchedSource,
    /// The requested node/argument/property could not be located.
    NotFound,
    /// The replacement text did not parse back into the expected value.
    InvalidReplacement,
    /// Two edits in the same plan touched overlapping byte ranges.
    OverlappingEdits,
    /// [`EditPlan::apply`] was given a source that was not the one the plan was
    /// built from.
    StaleSource,
    /// A computed byte range did not lie on a UTF-8 boundary.
    InvalidUtf8,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct EditErrorInner {
    message: String,
    label: Option<String>,
    help: Option<String>,
}

/// Errors that can happen while building or applying an [`EditPlan`].
///
/// Each error keeps the source string it relates to plus a [`SourceSpan`]
/// pointing at the offending bytes, so it implements [`Diagnostic`] and can be
/// pretty printed the same way parse errors are.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EditError {
    /// The source string the failing plan was built from.
    pub input: Arc<String>,
    /// What went wrong.
    pub kind: EditErrorKind,
    /// Primary span the error is about.
    pub span: SourceSpan,
    /// Optional second span (for example, the edit an earlier one collides
    /// with).
    pub other_span: Option<SourceSpan>,
    inner: Box<EditErrorInner>,
}

impl EditError {
    fn new(
        input: Arc<String>,
        kind: EditErrorKind,
        span: SourceSpan,
        other_span: Option<SourceSpan>,
        message: impl Into<String>,
        label: Option<impl Into<String>>,
        help: Option<impl Into<String>>,
    ) -> Self {
        Self {
            input,
            kind,
            span,
            other_span,
            inner: Box::new(EditErrorInner {
                message: message.into(),
                label: label.map(Into::into),
                help: help.map(Into::into),
            }),
        }
    }

    /// The human readable diagnostic message.
    pub fn message(&self) -> &str {
        &self.inner.message
    }

    /// The label for the primary span, if any.
    pub fn label(&self) -> Option<&str> {
        self.inner.label.as_deref()
    }

    /// The help text, if any.
    pub fn help(&self) -> Option<&str> {
        self.inner.help.as_deref()
    }
}

impl Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner.message)
    }
}

impl Error for EditError {}

impl Diagnostic for EditError {
    fn source_code(&self) -> Option<&dyn miette::SourceCode> {
        Some(&self.input)
    }

    fn severity(&self) -> Option<Severity> {
        Some(Severity::Error)
    }

    fn help<'a>(&'a self) -> Option<Box<dyn Display + 'a>> {
        self.inner
            .help
            .as_ref()
            .map(|s| Box::new(s) as Box<dyn Display>)
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        let label = self
            .inner
            .label
            .clone()
            .unwrap_or_else(|| "here".to_owned());
        let mut spans = vec![LabeledSpan::new_with_span(Some(label), self.span)];
        if let Some(other) = self.other_span {
            spans.push(LabeledSpan::new_with_span(
                Some("conflicts with this edit".to_owned()),
                other,
            ));
        }
        Some(Box::new(spans.into_iter()))
    }
}

/// A plan of non-overlapping, trivia-preserving edits to a single KDL source.
///
/// A plan is always created with [`EditPlan::new`], which binds it to the
/// exact source string the document was parsed from. High level operations
/// are added one at a time; each one is verified against that source. Call
/// [`EditPlan::edits`] to inspect the resulting [`SourceEdit`]s, or
/// [`EditPlan::apply`] to splice them into a copy of the source.
///
/// The plan borrows neither the document nor its nodes: it records byte spans
/// and rendered replacement text. Nodes are looked up by reference only at the
/// time an operation is added, which makes it safe to keep planning even after
/// the original tree is dropped or mutated.
#[derive(Debug, Clone)]
pub struct EditPlan {
    source: Arc<String>,
    source_hash: u64,
    edits: Vec<SourceEdit>,
}

impl EditPlan {
    /// Creates an empty plan bound to `source`.
    ///
    /// `source` must be the exact string the KDL document (and every node or
    /// entry passed to later operations) was parsed from.
    pub fn new(source: impl Into<Arc<String>>) -> Self {
        let source = source.into();
        let source_hash = fnv1a64(source.as_bytes());
        Self {
            source,
            source_hash,
            edits: Vec::new(),
        }
    }

    /// The source string the plan is bound to.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The edits recorded so far, ordered by their position in the source.
    pub fn edits(&self) -> &[SourceEdit] {
        &self.edits
    }

    /// Consumes the plan and returns its edits, ordered by position.
    pub fn into_edits(self) -> Vec<SourceEdit> {
        self.edits
    }

    /// Whether any edits have been recorded.
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Number of recorded edits.
    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// Rename a node, replacing only its name identifier token.
    ///
    /// The node's type annotation, arguments, properties, children and all
    /// surrounding trivia are left untouched. `name` is rendered the same way
    /// the tree renderer would render it (quoting it when necessary).
    pub fn set_node_name(
        &mut self,
        node: &KdlNode,
        name: impl Into<KdlIdentifier>,
    ) -> Result<&mut Self, EditError> {
        NodeBounds::compute(&self.source, node)?;
        let name = name.into();
        let rendered = name.to_string();
        validate_identifier(&self.source, node.name().span(), &rendered, &name)?;
        self.push(SourceEdit {
            kind: EditKind::NodeName,
            span: node.name().span(),
            replacement: rendered,
        })
    }

    /// Replace the value of a positional argument, indexed from zero (the same
    /// indexing used by [`KdlNode::get`]).
    ///
    /// Only the value literal itself is replaced. An existing type annotation,
    /// the whitespace before the entry and all comments are preserved.
    pub fn set_argument(
        &mut self,
        node: &KdlNode,
        index: usize,
        value: impl Into<KdlValue>,
    ) -> Result<&mut Self, EditError> {
        NodeBounds::compute(&self.source, node)?;
        let entry = node.entry(index).ok_or_else(|| {
            EditError::new(
                self.source.clone(),
                EditErrorKind::NotFound,
                node.name().span(),
                None,
                format!("Node has no argument at index {index}."),
                Some("node"),
                Some("Argument indices are zero-based and count only positional values."),
            )
        })?;
        self.set_entry_value(entry, value.into(), EditKind::Argument)
    }

    /// Replace the value of the property named `key`.
    ///
    /// As with the tree API, the *last* property with that name is the one
    /// addressed. Only the value literal is replaced; the key, `=`, type
    /// annotation and trivia are preserved.
    pub fn set_property(
        &mut self,
        node: &KdlNode,
        key: &str,
        value: impl Into<KdlValue>,
    ) -> Result<&mut Self, EditError> {
        NodeBounds::compute(&self.source, node)?;
        let entry = node.entry(key).ok_or_else(|| {
            EditError::new(
                self.source.clone(),
                EditErrorKind::NotFound,
                node.name().span(),
                None,
                format!("Node has no property named {key:?}."),
                Some("node"),
                Some(
                    "Only existing properties can be updated positionally; add new ones via the \
                     tree API.",
                ),
            )
        })?;
        self.set_entry_value(entry, value.into(), EditKind::Property)
    }

    /// Delete a node.
    ///
    /// The removed range is exactly what removing the node through the tree
    /// API would discard: the node's own leading trivia, the whole node and its
    /// terminator (including a trailing comment or a CRLF newline). The
    /// terminator that separates it from the following node and any document
    /// trailing trivia are preserved.
    pub fn delete_node(&mut self, node: &KdlNode) -> Result<&mut Self, EditError> {
        let bounds = NodeBounds::compute(&self.source, node)?;
        self.push(SourceEdit {
            kind: EditKind::DeleteNode,
            span: bounds.delete_span(),
            replacement: String::new(),
        })
    }

    fn set_entry_value(
        &mut self,
        entry: &KdlEntry,
        value: KdlValue,
        kind: EditKind,
    ) -> Result<&mut Self, EditError> {
        let rendered = value.to_string();
        let value_span = value_span(&self.source, entry)?;
        validate_value(&self.source, value_span, &rendered, &value)?;
        self.push(SourceEdit {
            kind,
            span: value_span,
            replacement: rendered,
        })
    }

    fn push(&mut self, edit: SourceEdit) -> Result<&mut Self, EditError> {
        for existing in &self.edits {
            if spans_overlap(existing.span, edit.span) {
                return Err(EditError::new(
                    self.source.clone(),
                    EditErrorKind::OverlappingEdits,
                    edit.span,
                    Some(existing.span),
                    format!(
                        "Refusing to apply overlapping edits ({} overlaps a previous {}).",
                        edit.kind, existing.kind
                    ),
                    Some("this edit"),
                    Some(
                        "Split the changes across multiple plans, or remove one of the edits. \
                         Containment and crossing edits are never reordered automatically.",
                    ),
                ));
            }
        }
        self.edits.push(edit);
        self.edits.sort_by_key(SourceEdit::start);
        Ok(self)
    }

    /// Applies the recorded edits to `source`, returning the new text.
    ///
    /// `source` must be identical (byte for byte) to the string the plan was
    /// created from. Its length and a checksum are verified up front, and every
    /// recorded span is re-checked while splicing, so applying a plan to a
    /// different version of the document fails with
    /// [`EditErrorKind::StaleSource`] / [`EditErrorKind::MismatchedSource`]
    /// instead of producing corrupt output.
    ///
    /// The resulting string can be re-parsed with
    /// [`KdlDocument::parse`](crate::KdlDocument::parse).
    pub fn apply(&self, source: &str) -> Result<String, EditError> {
        if source.len() != self.source.len() || fnv1a64(source.as_bytes()) != self.source_hash {
            return Err(EditError::new(
                self.source.clone(),
                EditErrorKind::StaleSource,
                (0..self.source.len()).into(),
                None,
                "Cannot apply an edit plan to a different source version than the one it was \
                 planned against.",
                Some("plan was built against this text"),
                Some("Re-parse the current source and rebuild the edit plan before applying it."),
            ));
        }
        apply_edits(source, &self.edits)
    }

    /// Applies the recorded edits to the plan's own bound source.
    pub fn apply_to_source(&self) -> Result<String, EditError> {
        apply_edits(&self.source, &self.edits)
    }
}

/// Applies a slice of non-overlapping [`SourceEdit`]s to `source`.
///
/// This is a convenience for callers that obtain edits from
/// [`EditPlan::edits`] (for example to hand them to another tool) and later
/// want to splice them back in. Edits must be non-overlapping and relative to
/// `source`; the spans are validated before splicing.
pub fn apply_edits(source: &str, edits: &[SourceEdit]) -> Result<String, EditError> {
    let arc = Arc::new(source.to_owned());
    let mut out = String::with_capacity(source.len());
    let mut cursor = 0usize;
    let mut last_end = 0usize;
    for edit in edits {
        let start = edit.start();
        let end = edit.end();
        if start < last_end || end < start || end > source.len() {
            return Err(EditError::new(
                arc.clone(),
                EditErrorKind::MismatchedSource,
                edit.span,
                None,
                "Edit span is out of bounds or overlaps another edit.",
                Some("invalid edit span"),
                Some("Only apply edits produced from this exact source."),
            ));
        }
        if !source.is_char_boundary(start) || !source.is_char_boundary(end) {
            return Err(EditError::new(
                arc.clone(),
                EditErrorKind::InvalidUtf8,
                edit.span,
                None,
                "Edit span does not lie on a UTF-8 boundary.",
                Some("invalid edit span"),
                None::<&str>,
            ));
        }
        out.push_str(&source[cursor..start]);
        out.push_str(&edit.replacement);
        cursor = end;
        last_end = end;
    }
    out.push_str(&source[cursor..]);
    Ok(out)
}

fn spans_overlap(a: SourceSpan, b: SourceSpan) -> bool {
    let a_start = a.offset();
    let a_end = a_start + a.len();
    let b_start = b.offset();
    let b_end = b_start + b.len();
    a_start < b_end && b_start < a_end
}

/// Resolves the byte span of an entry's value literal.
///
/// The entry span covers `key=value` (or just the value for an argument) and
/// never includes leading/trailing whitespace, and the parser stores the exact
/// source text of the value in `value_repr`. The value therefore occupies the
/// trailing `value_repr.len()` bytes of the entry span. This holds for plain,
/// quoted and raw strings (including multiline ones), numbers, keywords and
/// typed values.
fn value_span(source: &str, entry: &KdlEntry) -> Result<SourceSpan, EditError> {
    let arc = Arc::new(source.to_owned());
    let Some(fmt) = entry.format() else {
        return Err(EditError::new(
            arc,
            EditErrorKind::MismatchedSource,
            entry.span(),
            None,
            "Entry does not carry original source text, so it cannot be located.",
            Some("entry"),
            Some("Only entries obtained by parsing a document can be edited positionally."),
        ));
    };
    if fmt.value_repr.is_empty() {
        return Err(EditError::new(
            arc,
            EditErrorKind::MismatchedSource,
            entry.span(),
            None,
            "Entry has no recorded value representation, so it cannot be located.",
            Some("entry"),
            Some("Only entries obtained by parsing a document can be edited positionally."),
        ));
    }
    let repr = fmt.value_repr.as_str();
    let span = entry.span();
    if span.len() < repr.len() {
        return Err(mismatched(
            &arc,
            span,
            "Entry span is shorter than its recorded value representation.",
        ));
    }
    let start = span.offset() + span.len() - repr.len();
    let end = span.offset() + span.len();
    if end > source.len()
        || !source.is_char_boundary(start)
        || !source.is_char_boundary(end)
        || &source[start..end] != repr
    {
        return Err(mismatched(
            &arc,
            span,
            "Entry's recorded value does not match the source at its span.",
        ));
    }
    Ok((start..end).into())
}

/// The exact byte range a node occupies, including the trivia that the tree
/// renderer treats as belonging to that node.
#[derive(Debug, Clone, Copy)]
struct NodeBounds {
    leading_start: usize,
    body_start: usize,
    body_end: usize,
    terminator_end: usize,
}

impl NodeBounds {
    fn compute(source: &str, node: &KdlNode) -> Result<Self, EditError> {
        let arc = Arc::new(source.to_owned());
        let span = node.span();
        let body_start = span.offset();
        let body_end = body_start + span.len();
        if body_end > source.len()
            || !source.is_char_boundary(body_start)
            || !source.is_char_boundary(body_end)
        {
            return Err(mismatched(
                &arc,
                span,
                "Node span is outside the source bounds.",
            ));
        }

        let Some(fmt) = node.format() else {
            return Err(EditError::new(
                arc,
                EditErrorKind::MismatchedSource,
                span,
                None,
                "Node does not carry original formatting, so it cannot be located.",
                Some("node"),
                Some("Only nodes obtained by parsing a document can be edited positionally."),
            ));
        };
        let leading = fmt.leading.as_str();
        let terminator = fmt.terminator.as_str();

        let Some(leading_start) = body_start.checked_sub(leading.len()) else {
            return Err(mismatched(
                &arc,
                span,
                "Node's recorded leading trivia extends before the start of the source.",
            ));
        };
        let terminator_end = body_end + terminator.len();

        // Anchor verification: the span, leading trivia and terminator must
        // all point at the exact bytes recorded on the node. This is what
        // rejects nodes parsed from a different source or an older version.
        if !source.is_char_boundary(leading_start) || &source[leading_start..body_start] != leading
        {
            return Err(mismatched(
                &arc,
                span,
                "Node's leading trivia does not match the source at its span.",
            ));
        }
        if terminator_end > source.len()
            || !source.is_char_boundary(terminator_end)
            || &source[body_end..terminator_end] != terminator
        {
            return Err(mismatched(
                &arc,
                span,
                "Node's terminator does not match the source at its span.",
            ));
        }

        Ok(Self {
            leading_start,
            body_start,
            body_end,
            terminator_end,
        })
    }

    /// Range to remove when deleting the node: leading trivia + body +
    /// terminator.
    fn delete_span(self) -> SourceSpan {
        debug_assert!(self.leading_start <= self.body_start);
        debug_assert!(self.body_start <= self.body_end);
        debug_assert!(self.body_end <= self.terminator_end);
        (self.leading_start..self.terminator_end).into()
    }
}

fn mismatched(arc: &Arc<String>, span: SourceSpan, message: &str) -> EditError {
    EditError::new(
        arc.clone(),
        EditErrorKind::MismatchedSource,
        span,
        None,
        message,
        Some("recorded span"),
        Some(
            "This node or entry was parsed from a different source (or the source changed). \
             Re-parse and rebuild the plan.",
        ),
    )
}

fn validate_identifier(
    source: &str,
    span: SourceSpan,
    rendered: &str,
    expected: &KdlIdentifier,
) -> Result<(), EditError> {
    let arc = Arc::new(source.to_owned());
    let parsed = v2_parser::try_parse(v2_parser::identifier, rendered).map_err(|_| {
        EditError::new(
            arc.clone(),
            EditErrorKind::InvalidReplacement,
            span,
            None,
            format!("{rendered:?} is not a valid KDL identifier."),
            Some("node name"),
            Some("Use a value that renders as a valid KDL node name."),
        )
    })?;
    if parsed.value() != expected.value() {
        return Err(EditError::new(
            arc,
            EditErrorKind::InvalidReplacement,
            span,
            None,
            "Replacement node name did not round-trip to the expected value.",
            Some("node name"),
            None::<&str>,
        ));
    }
    Ok(())
}

fn validate_value(
    source: &str,
    span: SourceSpan,
    rendered: &str,
    expected: &KdlValue,
) -> Result<(), EditError> {
    let arc = Arc::new(source.to_owned());
    // A leading space makes this an unambiguous standalone entry for the
    // padded entry parser; trailing whitespace/newline is tolerated.
    let probe = format!(" {rendered}");
    let parsed =
        v2_parser::try_parse(v2_parser::padded_node_entry, probe.as_str()).map_err(|_| {
            EditError::new(
                arc.clone(),
                EditErrorKind::InvalidReplacement,
                span,
                None,
                format!("{rendered:?} is not a valid KDL value."),
                Some("replacement value"),
                None::<&str>,
            )
        })?;
    if parsed.value() != expected {
        return Err(EditError::new(
            arc,
            EditErrorKind::InvalidReplacement,
            span,
            None,
            "Replacement value did not round-trip to the expected value.",
            Some("replacement value"),
            None::<&str>,
        ));
    }
    Ok(())
}

/// Deterministic, non-cryptographic checksum used to reject applying a plan to
/// a source that merely has the same length as the original.
fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
