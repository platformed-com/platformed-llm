//! Helper for provider stream conversions to allocate part indices.
//!
//! Each provider's wire stream identifies parts by a provider-specific key:
//! Anthropic uses `u32` indices; OpenAI uses `(output_index,
//! content_index)`; Gemini synthesizes. `PartTracker<K>` maps those keys
//! to our lib-side monotonically-increasing `u32` part indices.

use std::collections::HashMap;
use std::hash::Hash;

use crate::types::{PartKind, StreamEvent};

#[derive(Debug, Default)]
pub(crate) struct PartTracker<K: Eq + Hash> {
    next_index: u32,
    by_key: HashMap<K, u32>,
}

// Per-method dead-code allow: each provider uses a different subset
// (Anthropic doesn't need `new` / `open_one_shot`, etc.), so the
// "unused" set varies by feature combination. The allow keeps the
// shared impl block warning-free under every gate combo.
#[allow(dead_code)]
impl<K: Eq + Hash + Clone> PartTracker<K> {
    pub fn new() -> Self {
        Self {
            next_index: 0,
            by_key: HashMap::new(),
        }
    }

    /// Open a new part. Returns the assigned index and the `PartStart`
    /// event to emit.
    pub fn open(&mut self, key: K, kind: PartKind) -> (u32, StreamEvent) {
        let index = self.next_index;
        self.next_index += 1;
        self.by_key.insert(key, index);
        (index, StreamEvent::PartStart { index, kind })
    }

    pub fn index_of(&self, key: &K) -> Option<u32> {
        self.by_key.get(key).copied()
    }

    /// Allocate a part index without binding it to a key, then emit
    /// the PartStart + PartEnd pair atomically. Use for one-shot parts
    /// whose payload is fully carried in `kind` (e.g.
    /// [`PartKind::Continuation`], [`PartKind::RedactedReasoning`]).
    pub fn open_one_shot(&mut self, kind: PartKind) -> Vec<StreamEvent> {
        let index = self.next_index;
        self.next_index += 1;
        vec![
            StreamEvent::PartStart { index, kind },
            StreamEvent::PartEnd { index },
        ]
    }

    /// Close a previously-opened part. Returns the `PartEnd` event or
    /// `None` if the key wasn't open.
    pub fn close(&mut self, key: &K) -> Option<StreamEvent> {
        let index = self.by_key.remove(key)?;
        Some(StreamEvent::PartEnd { index })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_assigns_monotonic_indices() {
        let mut t: PartTracker<u32> = PartTracker::new();
        let (i0, _) = t.open(10, PartKind::Text);
        let (i1, _) = t.open(20, PartKind::Reasoning);
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(t.index_of(&10), Some(0));
    }

    #[test]
    fn close_emits_part_end() {
        let mut t: PartTracker<u32> = PartTracker::new();
        t.open(42, PartKind::Text);
        let end = t.close(&42).expect("found");
        assert!(matches!(end, StreamEvent::PartEnd { index: 0 }));
        assert_eq!(t.index_of(&42), None);
    }

    #[test]
    fn close_unknown_returns_none() {
        let mut t: PartTracker<u32> = PartTracker::new();
        assert!(t.close(&999).is_none());
    }
}
