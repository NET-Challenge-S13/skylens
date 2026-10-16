//! Small DOM helpers over a Blitz document.
//!
//! The screens update the legacy DOM the way the TypeScript did: toggle classes like `is-hidden`,
//! set text, set attributes and inline styles, replace an element's children with markup. Writes
//! that would not change anything are skipped, because every write restyles and relays out the
//! subtree. [`Dom::changed`] tells the shell whether to ask for a redraw.

use blitz_dom::{Attribute, BaseDocument, LocalName, Namespace, NodeId, QualName, ns};
use std::collections::HashMap;

/// How an element is addressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sel<'a> {
    /// `id` attribute.
    Id(&'a str),
    /// A node already found.
    Node(NodeId),
    /// First match of a CSS selector.
    Query(&'a str),
}

impl From<NodeId> for Sel<'_> {
    fn from(n: NodeId) -> Self {
        Sel::Node(n)
    }
}

/// A bare string is an element id.
impl<'a> From<&'a str> for Sel<'a> {
    fn from(id: &'a str) -> Self {
        Sel::Id(id)
    }
}

impl<'a> From<&'a String> for Sel<'a> {
    fn from(id: &'a String) -> Self {
        Sel::Id(id.as_str())
    }
}

/// A CSS selector (first match), e.g. `q(".detect-card__label")`.
pub fn q(selector: &str) -> Sel<'_> {
    Sel::Query(selector)
}

/// Escapes text for use inside markup.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Splits a `class` attribute and adds or removes one class. Returns the new value, or None when
/// nothing changes.
pub fn toggled_class_list(current: &str, class: &str, on: bool) -> Option<String> {
    let mut classes: Vec<&str> = current.split_whitespace().collect();
    let has = classes.contains(&class);
    match (has, on) {
        (true, true) | (false, false) => None,
        (false, true) => {
            classes.push(class);
            Some(classes.join(" "))
        }
        (true, false) => {
            classes.retain(|c| *c != class);
            Some(classes.join(" "))
        }
    }
}

fn attr_name(name: &str) -> QualName {
    QualName::new(None, Namespace::from(""), LocalName::from(name))
}

fn element_name(tag: &str) -> QualName {
    QualName::new(None, ns!(html), LocalName::from(tag))
}

/// Remembers the markup last written into each node, so replacing children with the same markup
/// is free. Owned by the shell and lent to every [`Dom`].
#[derive(Debug, Default)]
pub struct MarkupCache {
    html: HashMap<NodeId, String>,
}

impl MarkupCache {
    pub fn forget(&mut self, node: NodeId) {
        self.html.remove(&node);
    }
}

/// Mutable view of the document for one update.
pub struct Dom<'a> {
    doc: &'a mut BaseDocument,
    cache: &'a mut MarkupCache,
    changed: bool,
}

impl<'a> Dom<'a> {
    pub fn new(doc: &'a mut BaseDocument, cache: &'a mut MarkupCache) -> Dom<'a> {
        Dom { doc, cache, changed: false }
    }

    /// Whether any write went through.
    pub fn changed(&self) -> bool {
        self.changed
    }

    /// Marks the view as changed (the screen changed something the DOM does not show, e.g. 3D).
    pub fn mark_changed(&mut self) {
        self.changed = true;
    }

    /// The underlying Blitz document, for anything these helpers do not cover.
    pub fn doc(&mut self) -> &mut BaseDocument {
        self.doc
    }

    pub fn doc_ref(&self) -> &BaseDocument {
        self.doc
    }

    // ---- finding --------------------------------------------------------------------------

    pub fn find<'s>(&self, sel: impl Into<Sel<'s>>) -> Option<NodeId> {
        match sel.into() {
            Sel::Id(id) => self.doc.get_element_by_id(id),
            Sel::Node(n) => self.doc.get_node(n).map(|_| n),
            Sel::Query(s) => self.doc.query_selector(s).ok().flatten(),
        }
    }

    /// Every match of a CSS selector, in document order.
    pub fn find_all(&self, selector: &str) -> Vec<NodeId> {
        self.doc.query_selector_all(selector).map(|v| v.to_vec()).unwrap_or_default()
    }

    /// First match of `selector` inside `scope`.
    pub fn find_in<'s>(&self, scope: impl Into<Sel<'s>>, selector: &str) -> Option<NodeId> {
        let scope = self.find(scope)?;
        self.doc.query_selector_in(scope, selector).ok().flatten()
    }

    pub fn exists<'s>(&self, sel: impl Into<Sel<'s>>) -> bool {
        self.find(sel).is_some()
    }

    /// Element children of a node.
    pub fn children<'s>(&self, sel: impl Into<Sel<'s>>) -> Vec<NodeId> {
        let Some(n) = self.find(sel) else { return Vec::new() };
        self.doc
            .get_node(n)
            .map(|node| {
                node.children
                    .iter()
                    .copied()
                    .filter(|c| self.doc.get_node(*c).is_some_and(|c| c.is_element()))
                    .collect()
            })
            .unwrap_or_default()
    }

    // ---- reading --------------------------------------------------------------------------

    pub fn text<'s>(&self, sel: impl Into<Sel<'s>>) -> Option<String> {
        let n = self.find(sel)?;
        self.doc.get_node(n).map(|node| node.text_content())
    }

    pub fn attr<'s>(&self, sel: impl Into<Sel<'s>>, name: &str) -> Option<String> {
        let n = self.find(sel)?;
        self.doc.get_node(n)?.element_data()?.attr(LocalName::from(name)).map(str::to_string)
    }

    pub fn has_class<'s>(&self, sel: impl Into<Sel<'s>>, class: &str) -> bool {
        self.attr(sel, "class").is_some_and(|c| c.split_whitespace().any(|x| x == class))
    }

    /// Current value of a text input (what the user typed), else its `value` attribute.
    pub fn input_value<'s>(&self, sel: impl Into<Sel<'s>>) -> Option<String> {
        let n = self.find(sel)?;
        let el = self.doc.get_node(n)?.element_data()?;
        if let Some(t) = el.text_input_data() {
            return Some(t.editor.raw_text().to_string());
        }
        el.attr(LocalName::from("value")).map(str::to_string)
    }

    /// Whether a checkbox is checked.
    pub fn checked<'s>(&self, sel: impl Into<Sel<'s>>) -> bool {
        self.find(sel)
            .and_then(|n| self.doc.get_node(n))
            .and_then(|n| n.element_data())
            .and_then(|e| e.checkbox_input_checked())
            .unwrap_or(false)
    }

    // ---- writing --------------------------------------------------------------------------

    /// Replaces the node's content with one text node (like `textContent =`).
    pub fn set_text<'s>(&mut self, sel: impl Into<Sel<'s>>, text: &str) -> bool {
        let Some(n) = self.find(sel) else { return false };
        let Some(node) = self.doc.get_node(n) else { return false };
        // One text child already: change it in place.
        if node.children.len() == 1 {
            let child = node.children[0];
            if let Some(t) = self.doc.get_node(child).and_then(|c| c.text_data()) {
                if t.content == text {
                    return false;
                }
                self.doc.mutate().set_node_text(child, text);
                self.cache.forget(n);
                self.changed = true;
                return true;
            }
        }
        if node.children.is_empty() && text.is_empty() {
            return false;
        }
        {
            let mut m = self.doc.mutate();
            m.remove_and_drop_all_children(n);
            if !text.is_empty() {
                let t = m.create_text_node(text);
                m.append_children(n, &[t]);
            }
        }
        self.cache.forget(n);
        self.changed = true;
        true
    }

    /// Replaces the node's children with parsed markup (like `innerHTML =`). Skipped when the
    /// same markup was written last time.
    pub fn set_html<'s>(&mut self, sel: impl Into<Sel<'s>>, markup: &str) -> bool {
        let Some(n) = self.find(sel) else { return false };
        if self.cache.html.get(&n).is_some_and(|m| m == markup) {
            return false;
        }
        self.doc.mutate().set_inner_html(n, markup);
        self.cache.html.insert(n, markup.to_string());
        self.changed = true;
        true
    }

    /// Same as [`Self::set_html`]: the legacy `replaceChildren(...)` with markup.
    pub fn replace_children<'s>(&mut self, sel: impl Into<Sel<'s>>, markup: &str) -> bool {
        self.set_html(sel, markup)
    }

    /// Appends parsed markup after the node's children. Returns the new top-level nodes.
    pub fn append_html<'s>(&mut self, sel: impl Into<Sel<'s>>, markup: &str) -> Vec<NodeId> {
        let Some(parent) = self.find(sel) else { return Vec::new() };
        // Parse into a detached holder, then move its children over.
        let holder = {
            let mut m = self.doc.mutate();
            let holder = m.create_element(element_name("div"), Vec::new());
            m.set_inner_html(holder, markup);
            holder
        };
        let added: Vec<NodeId> = self
            .doc
            .get_node(holder)
            .map(|h| h.children.iter().copied().collect())
            .unwrap_or_default();
        {
            let mut m = self.doc.mutate();
            m.append_children(parent, &added);
            m.remove_and_drop_node(holder);
        }
        self.cache.forget(parent);
        self.changed = true;
        added
    }

    /// Creates `<tag attrs...>text</tag>` as the last child of `parent`.
    pub fn create<'s>(
        &mut self,
        parent: impl Into<Sel<'s>>,
        tag: &str,
        attrs: &[(&str, &str)],
        text: Option<&str>,
    ) -> Option<NodeId> {
        let parent = self.find(parent)?;
        let id = {
            let mut m = self.doc.mutate();
            let attrs = attrs
                .iter()
                .map(|(k, v)| Attribute { name: attr_name(k), value: (*v).to_string() })
                .collect();
            let el = m.create_element(element_name(tag), attrs);
            if let Some(t) = text.filter(|t| !t.is_empty()) {
                let t = m.create_text_node(t);
                m.append_children(el, &[t]);
            }
            m.append_children(parent, &[el]);
            el
        };
        self.cache.forget(parent);
        self.changed = true;
        Some(id)
    }

    /// Removes the node (and its subtree) from the document.
    pub fn remove<'s>(&mut self, sel: impl Into<Sel<'s>>) -> bool {
        let Some(n) = self.find(sel) else { return false };
        let parent = self.doc.get_node(n).and_then(|node| node.parent);
        self.doc.mutate().remove_and_drop_node(n);
        self.cache.forget(n);
        if let Some(p) = parent {
            self.cache.forget(p);
        }
        self.changed = true;
        true
    }

    pub fn set_attr<'s>(&mut self, sel: impl Into<Sel<'s>>, name: &str, value: &str) -> bool {
        let Some(n) = self.find(sel) else { return false };
        if self.attr(n, name).as_deref() == Some(value) {
            return false;
        }
        self.doc.mutate().set_attribute(n, attr_name(name), value);
        self.changed = true;
        true
    }

    pub fn remove_attr<'s>(&mut self, sel: impl Into<Sel<'s>>, name: &str) -> bool {
        let Some(n) = self.find(sel) else { return false };
        if self.attr(n, name).is_none() {
            return false;
        }
        self.doc.mutate().clear_attribute(n, attr_name(name));
        self.changed = true;
        true
    }

    /// Adds (`on`) or removes a class, like `classList.toggle(class, on)`.
    pub fn toggle_class<'s>(&mut self, sel: impl Into<Sel<'s>>, class: &str, on: bool) -> bool {
        let Some(n) = self.find(sel) else { return false };
        let current = self.attr(n, "class").unwrap_or_default();
        match toggled_class_list(&current, class, on) {
            Some(next) => {
                self.doc.mutate().set_attribute(n, attr_name("class"), &next);
                self.changed = true;
                true
            }
            None => false,
        }
    }

    pub fn add_class<'s>(&mut self, sel: impl Into<Sel<'s>>, class: &str) -> bool {
        self.toggle_class(sel, class, true)
    }

    pub fn remove_class<'s>(&mut self, sel: impl Into<Sel<'s>>, class: &str) -> bool {
        self.toggle_class(sel, class, false)
    }

    /// Replaces the whole `class` attribute.
    pub fn set_class<'s>(&mut self, sel: impl Into<Sel<'s>>, classes: &str) -> bool {
        self.set_attr(sel, "class", classes)
    }

    /// Sets one inline style property, like `el.style.width = '40%'`. Blitz keeps inline styles
    /// apart from the `style` attribute, so this is not deduplicated; call it on change.
    pub fn set_style<'s>(&mut self, sel: impl Into<Sel<'s>>, property: &str, value: &str) -> bool {
        let Some(n) = self.find(sel) else { return false };
        self.doc.mutate().set_style_property(n, property, value);
        self.changed = true;
        true
    }

    pub fn remove_style<'s>(&mut self, sel: impl Into<Sel<'s>>, property: &str) -> bool {
        let Some(n) = self.find(sel) else { return false };
        self.doc.mutate().remove_style_property(n, property);
        self.changed = true;
        true
    }

    /// Mounts a custom widget (the 3D view) on an element, e.g. the legacy `<canvas>`.
    pub fn mount_widget<'s>(&mut self, sel: impl Into<Sel<'s>>, widget: Box<dyn blitz_dom::Widget>) -> bool {
        let Some(n) = self.find(sel) else { return false };
        self.doc.mutate().set_custom_widget(n, widget);
        self.changed = true;
        true
    }

    /// Scrolls a scrollable element (`overflow: auto`) to an absolute offset in CSS pixels, at once.
    pub fn scroll_to<'s>(&mut self, sel: impl Into<Sel<'s>>, x: f64, y: f64) -> bool {
        let Some(n) = self.find(sel) else { return false };
        self.doc.scroll_to(n, x, y, blitz_dom::ScrollBehavior::Instant);
        self.changed = true;
        true
    }

    /// The element's laid-out size in CSS pixels, once it has been laid out.
    pub fn size_of<'s>(&self, sel: impl Into<Sel<'s>>) -> Option<(f64, f64)> {
        let n = self.find(sel)?;
        let l = self.doc.get_node(n)?.final_layout();
        Some((f64::from(l.size.width), f64::from(l.size.height)))
    }

    /// Moves keyboard focus to an element.
    pub fn focus<'s>(&mut self, sel: impl Into<Sel<'s>>) -> bool {
        let Some(n) = self.find(sel) else { return false };
        self.doc.set_focus_to(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_markup() {
        assert_eq!(escape("a<b>&\"c'"), "a&lt;b&gt;&amp;&quot;c&#39;");
        assert_eq!(escape("구간 1 · 정제"), "구간 1 · 정제");
    }

    #[test]
    fn toggles_classes_like_class_list() {
        assert_eq!(toggled_class_list("loading-screen", "is-hidden", true).as_deref(), Some("loading-screen is-hidden"));
        assert_eq!(toggled_class_list("a is-hidden b", "is-hidden", false).as_deref(), Some("a b"));
        assert_eq!(toggled_class_list("a is-hidden", "is-hidden", true), None);
        assert_eq!(toggled_class_list("a", "is-hidden", false), None);
        assert_eq!(toggled_class_list("", "x", true).as_deref(), Some("x"));
        // A class that is a prefix of another is not confused with it.
        assert_eq!(toggled_class_list("is-hidden-soon", "is-hidden", true).as_deref(), Some("is-hidden-soon is-hidden"));
    }
}
