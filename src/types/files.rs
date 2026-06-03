//! Caller-held file registry: portable file references resolved to
//! provider-specific handles at request time.
//!
//! The conversation history references binary inputs only by a
//! **caller-opaque ID** ([`UserPart::Image`](super::UserPart) etc. carrying
//! an `FileSource::Ref`). All knowledge of provider-specific upload handles
//! lives behind a caller-supplied [`FileResolver`] — the "registry". The
//! library never persists this state itself; it asks the resolver to
//! [`lookup`](FileResolver::lookup) an existing handle, [`open`](FileResolver::open)
//! a fresh byte stream when one must be uploaded, and [`store`](FileResolver::store)
//! the handle it just created.
//!
//! This keeps two invariants intact:
//! - **History stays portable.** Only the opaque ID is ever written into an
//!   [`InputItem`](super::InputItem); provider handles exist only transiently
//!   during request-building, so switching providers mid-conversation still
//!   works.
//! - **The library stays stateless.** The caller owns persistence and
//!   lifecycle; the resolver is the source of truth, the handles it returns
//!   are a cache.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;
use serde::{Deserialize, Serialize};

use crate::factory::ProviderType;
use crate::Error;

/// Current Unix time in whole seconds (UTC). Saturates to 0 before the epoch.
pub(crate) fn now_unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// The identity within which a provider file handle is valid.
///
/// A handle is **not** portable across providers, and often not even across
/// accounts of the same provider (an OpenAI file ID is org-scoped; a Vertex
/// file is project- and region-scoped). The resolver is keyed on the full
/// scope so a handle minted for one account is never handed to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderScope {
    /// Which backend the handle targets.
    pub provider: ProviderType,
    /// Account / project+region / org identity the handle lives within.
    /// Built by the provider from its own configuration (e.g.
    /// `"project/location"` for Vertex, the API base for OpenAI).
    pub account: String,
}

impl ProviderScope {
    /// Build a scope from its parts.
    pub fn new(provider: ProviderType, account: impl Into<String>) -> Self {
        Self {
            provider,
            account: account.into(),
        }
    }
}

/// A provider-ready file reference the library can drop straight into a
/// wire request (an OpenAI file ID, an Anthropic file ID, a `gs://` URI, …).
///
/// Derives `Serialize`/`Deserialize` so the registry round-trips cleanly
/// through a database — the caller persists these and hands them back from
/// [`FileResolver::lookup`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedHandle {
    /// The provider-specific reference — file ID, `gs://` URI, etc.
    pub uri: String,
    /// MIME type of the referenced file (e.g. `application/pdf`).
    pub media_type: String,
    /// Unix timestamp (seconds since the epoch, UTC) at which the handle
    /// expires, if ever. `None` means "no known expiry" (e.g. a `gs://` URI
    /// you own, an OpenAI file). A plain integer — not a `SystemTime` — so it
    /// stores and reloads from a database without conversion. The library
    /// treats it as a **hint**: it refreshes within a safety margin, but also
    /// recovers from a provider "not found" because a handle can die before
    /// its stated expiry.
    pub expires_at: Option<i64>,
}

impl ResolvedHandle {
    /// An **OpenAI** file handle — the `file-…` id returned by
    /// `POST /v1/files`. Referenced on the wire as `file_id`. No known expiry
    /// (OpenAI files persist until deleted); add one with [`Self::with_expiry`]
    /// if you GC them on a schedule.
    pub fn openai_file(file_id: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            uri: file_id.into(),
            media_type: media_type.into(),
            expires_at: None,
        }
    }

    /// A **Gemini (Vertex)** file handle backed by a Cloud Storage object —
    /// a `gs://bucket/object` URI. Referenced on the wire as
    /// `fileData.fileUri`. Durable for as long as the object lives in your
    /// bucket, so no expiry by default.
    pub fn gemini_gcs(gcs_uri: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            uri: gcs_uri.into(),
            media_type: media_type.into(),
            expires_at: None,
        }
    }

    /// An **Anthropic** file handle — the `file_id` from the Anthropic Files
    /// API. Referenced on the wire as a `{"type":"file","file_id":…}` source.
    pub fn anthropic_file(file_id: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            uri: file_id.into(),
            media_type: media_type.into(),
            expires_at: None,
        }
    }

    /// Set an expiry (Unix seconds since the epoch, UTC) on this handle. See
    /// [`Self::expires_at`].
    pub fn with_expiry(mut self, expires_at_unix_secs: i64) -> Self {
        self.expires_at = Some(expires_at_unix_secs);
        self
    }

    /// Wrap this handle as a [`ResolvedFile::ProviderHandle`] — the form
    /// [`FileResolver::open`] returns. [`FileResolver::lookup`] returns the
    /// bare [`ResolvedHandle`] instead.
    pub fn into_file(self) -> ResolvedFile {
        ResolvedFile::ProviderHandle(self)
    }

    /// Whether this handle is expired as of `now_unix_secs` (no safety
    /// margin). A handle with no stated expiry is never expired.
    pub fn is_expired(&self, now_unix_secs: i64) -> bool {
        matches!(self.expires_at, Some(exp) if exp <= now_unix_secs)
    }
}

/// What a [`FileResolver::open`] call hands back: the bytes to upload, or a
/// reference the library can use without uploading.
///
/// This is **not** `Clone` — the `Stream` variant is single-use. The library
/// drives retries by calling [`FileResolver::open`] again for a fresh stream,
/// never by replaying a consumed body.
pub enum ResolvedFile {
    /// A fresh, independently-readable byte stream for the library to upload.
    /// Each [`FileResolver::open`] call must yield a new one.
    Stream {
        /// MIME type of the file.
        media_type: String,
        /// Total byte length when known. A known length unlocks the
        /// efficient single-shot / resumable upload path; `None` forces a
        /// chunked (degraded) upload, which some endpoints reject.
        content_length: Option<u64>,
        /// The byte stream. Mirrors
        /// [`TransportResponse::body`](crate::transport::TransportResponse::body);
        /// dropping it mid-upload must terminate cleanly.
        body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>,
    },
    /// The caller already holds a provider-specific reference for this
    /// `(id, scope)` — a provider file ID, or a provider-scoped URI such as a
    /// `gs://` object for Vertex. Referenced verbatim, no upload; the caller
    /// owns its lifecycle. Use this — **not** [`Url`](Self::Url) — for anything
    /// that isn't a plain public HTTP(S) URL.
    ProviderHandle(ResolvedHandle),
    /// A genuinely **public** HTTP(S) URL the provider fetches itself. For
    /// provider-specific URIs (e.g. a `gs://` object, a provider file ID) use
    /// [`ProviderHandle`](Self::ProviderHandle) instead — those are not public
    /// URLs and each provider references them with a different wire shape.
    Url {
        /// The public URL.
        uri: String,
        /// MIME type of the file the URL points at.
        media_type: String,
    },
}

impl ResolvedFile {
    /// A provider handle for **OpenAI** — see [`ResolvedHandle::openai_file`].
    /// Convenience for returning a handle from [`FileResolver::open`].
    pub fn openai_file(file_id: impl Into<String>, media_type: impl Into<String>) -> Self {
        ResolvedHandle::openai_file(file_id, media_type).into_file()
    }

    /// A provider handle for **Gemini (Vertex)** backed by a Cloud Storage
    /// object — see [`ResolvedHandle::gemini_gcs`].
    pub fn gemini_gcs(gcs_uri: impl Into<String>, media_type: impl Into<String>) -> Self {
        ResolvedHandle::gemini_gcs(gcs_uri, media_type).into_file()
    }

    /// A provider handle for **Anthropic** — see
    /// [`ResolvedHandle::anthropic_file`].
    pub fn anthropic_file(file_id: impl Into<String>, media_type: impl Into<String>) -> Self {
        ResolvedHandle::anthropic_file(file_id, media_type).into_file()
    }
}

impl std::fmt::Debug for ResolvedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolvedFile::Stream {
                media_type,
                content_length,
                ..
            } => f
                .debug_struct("ResolvedFile::Stream")
                .field("media_type", media_type)
                .field("content_length", content_length)
                .finish_non_exhaustive(),
            ResolvedFile::ProviderHandle(h) => f
                .debug_tuple("ResolvedFile::ProviderHandle")
                .field(h)
                .finish(),
            ResolvedFile::Url { uri, media_type } => f
                .debug_struct("ResolvedFile::Url")
                .field("uri", uri)
                .field("media_type", media_type)
                .finish(),
        }
    }
}

/// Caller-supplied indirection between portable file IDs and
/// provider-specific upload handles — the "file registry".
///
/// The library calls this during request-building for every
/// [`FileSource::Ref`](crate::FileSource::Ref) in the prompt. Implementors own
/// all persistence: a `lookup` hit lets the library skip uploading; a miss
/// triggers `open` → upload → `store`.
///
/// **`open` is a factory.** The library may call it more than once for the
/// same `(id, scope)` — on a transient upload failure, on recovery from an
/// expired/evicted handle, or on a redirect — and each call **must** yield an
/// independently readable payload.
#[async_trait]
pub trait FileResolver: Send + Sync {
    /// Return a cached provider handle for `(id, scope)` if the caller has
    /// one. Returning `None` (or an expired handle) makes the library
    /// `open` and upload.
    async fn lookup(
        &self,
        id: &str,
        scope: &ProviderScope,
    ) -> Result<Option<ResolvedHandle>, Error>;

    /// Produce the file for `(id, scope)`. May be called repeatedly; each
    /// call must yield a fresh, independent [`ResolvedFile`].
    async fn open(&self, id: &str, scope: &ProviderScope) -> Result<ResolvedFile, Error>;

    /// Persist a handle the library just uploaded so future `lookup` calls
    /// for `(id, scope)` hit it.
    async fn store(
        &self,
        id: &str,
        scope: &ProviderScope,
        handle: ResolvedHandle,
    ) -> Result<(), Error>;
}

/// An in-memory LRU cache in front of another [`FileResolver`].
///
/// Wrap your durable registry (a database, say) so repeated turns in a
/// conversation don't hit the backing store on every `lookup`:
///
/// - [`lookup`](FileResolver::lookup) checks the in-memory cache first; a
///   fresh hit returns immediately, a miss (or an expired entry) forwards to
///   the inner resolver and populates the cache.
/// - [`store`](FileResolver::store) writes through to **both** the cache and
///   the inner resolver.
/// - [`open`](FileResolver::open) always forwards — streams aren't cacheable.
///
/// Expired entries are evicted on access; the least-recently-used entry is
/// evicted when the cache is full.
pub struct LruFileResolver {
    inner: Arc<dyn FileResolver>,
    cache: Mutex<LruCache>,
}

impl LruFileResolver {
    /// Wrap `inner`, caching up to `capacity` handles in memory. A `capacity`
    /// of 0 disables caching (every call forwards to `inner`).
    pub fn new(inner: Arc<dyn FileResolver>, capacity: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(capacity)),
        }
    }
}

impl std::fmt::Debug for LruFileResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LruFileResolver").finish_non_exhaustive()
    }
}

/// Cache key — a handle is only valid within its `(provider, account)` scope,
/// so the id alone is not enough.
type CacheKey = (ProviderType, String, String);

struct LruCache {
    capacity: usize,
    /// Monotonic access counter; the entry with the lowest value is the LRU.
    tick: u64,
    entries: HashMap<CacheKey, (ResolvedHandle, u64)>,
}

impl LruCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            tick: 0,
            entries: HashMap::new(),
        }
    }

    fn key(scope: &ProviderScope, id: &str) -> CacheKey {
        (
            scope.provider.clone(),
            scope.account.clone(),
            id.to_string(),
        )
    }

    /// Fetch and bump recency.
    fn get(&mut self, scope: &ProviderScope, id: &str) -> Option<ResolvedHandle> {
        self.tick += 1;
        let tick = self.tick;
        let entry = self.entries.get_mut(&Self::key(scope, id))?;
        entry.1 = tick;
        Some(entry.0.clone())
    }

    fn remove(&mut self, scope: &ProviderScope, id: &str) {
        self.entries.remove(&Self::key(scope, id));
    }

    fn put(&mut self, scope: &ProviderScope, id: &str, handle: ResolvedHandle) {
        if self.capacity == 0 {
            return;
        }
        self.tick += 1;
        let tick = self.tick;
        self.entries.insert(Self::key(scope, id), (handle, tick));
        // Eviction scans for the lowest tick — O(n) per over-capacity insert.
        // Intentional: this cache is meant for small `capacity` (a conversation's
        // handful of files), where a linear scan beats the bookkeeping of an
        // intrusive linked list. Revisit if a large-capacity use case appears.
        while self.entries.len() > self.capacity {
            let Some(lru) = self
                .entries
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            self.entries.remove(&lru);
        }
    }
}

#[async_trait]
impl FileResolver for LruFileResolver {
    async fn lookup(
        &self,
        id: &str,
        scope: &ProviderScope,
    ) -> Result<Option<ResolvedHandle>, Error> {
        let now = now_unix_secs();
        // Cache probe — guard dropped before any await.
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(handle) = cache.get(scope, id) {
                if !handle.is_expired(now) {
                    return Ok(Some(handle));
                }
                cache.remove(scope, id);
            }
        }
        let result = self.inner.lookup(id, scope).await?;
        if let Some(handle) = &result {
            self.cache.lock().unwrap().put(scope, id, handle.clone());
        }
        Ok(result)
    }

    async fn open(&self, id: &str, scope: &ProviderScope) -> Result<ResolvedFile, Error> {
        self.inner.open(id, scope).await
    }

    async fn store(
        &self,
        id: &str,
        scope: &ProviderScope,
        handle: ResolvedHandle,
    ) -> Result<(), Error> {
        self.cache.lock().unwrap().put(scope, id, handle.clone());
        self.inner.store(id, scope, handle).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counting inner resolver: records how many times each method is hit so
    /// the cache's forwarding behaviour is observable.
    #[derive(Default)]
    struct Counting {
        lookups: AtomicUsize,
        stores: AtomicUsize,
        handle: Mutex<Option<ResolvedHandle>>,
    }

    #[async_trait]
    impl FileResolver for Counting {
        async fn lookup(
            &self,
            _id: &str,
            _scope: &ProviderScope,
        ) -> Result<Option<ResolvedHandle>, Error> {
            self.lookups.fetch_add(1, Ordering::SeqCst);
            Ok(self.handle.lock().unwrap().clone())
        }
        async fn open(&self, _id: &str, _scope: &ProviderScope) -> Result<ResolvedFile, Error> {
            Err(Error::config("not used"))
        }
        async fn store(
            &self,
            _id: &str,
            _scope: &ProviderScope,
            handle: ResolvedHandle,
        ) -> Result<(), Error> {
            self.stores.fetch_add(1, Ordering::SeqCst);
            *self.handle.lock().unwrap() = Some(handle);
            Ok(())
        }
    }

    fn scope() -> ProviderScope {
        ProviderScope::new(ProviderType::OpenAI, "acct")
    }

    fn handle(uri: &str, expires_at: Option<i64>) -> ResolvedHandle {
        ResolvedHandle {
            uri: uri.into(),
            media_type: "application/pdf".into(),
            expires_at,
        }
    }

    #[test]
    fn provider_handle_constructors_build_expected_shapes() {
        let h = ResolvedHandle::openai_file("file-123", "application/pdf");
        assert_eq!(h.uri, "file-123");
        assert_eq!(h.media_type, "application/pdf");
        assert_eq!(h.expires_at, None);

        let g = ResolvedHandle::gemini_gcs("gs://bucket/x.pdf", "application/pdf").with_expiry(42);
        assert_eq!(g.uri, "gs://bucket/x.pdf");
        assert_eq!(g.expires_at, Some(42));

        let a = ResolvedHandle::anthropic_file("file-abc", "image/png");
        assert_eq!(a.media_type, "image/png");

        // ResolvedFile wrappers and into_file() both yield a ProviderHandle.
        assert!(matches!(
            ResolvedFile::openai_file("file-1", "application/pdf"),
            ResolvedFile::ProviderHandle(h) if h.uri == "file-1"
        ));
        assert!(matches!(h.into_file(), ResolvedFile::ProviderHandle(_)));
    }

    #[tokio::test]
    async fn lookup_caches_after_first_inner_hit() {
        let inner = Arc::new(Counting::default());
        *inner.handle.lock().unwrap() = Some(handle("file-1", None));
        let lru = LruFileResolver::new(inner.clone(), 8);

        let a = lru.lookup("a", &scope()).await.unwrap();
        let b = lru.lookup("a", &scope()).await.unwrap();
        assert_eq!(a.unwrap().uri, "file-1");
        assert_eq!(b.unwrap().uri, "file-1");
        // Second lookup served from cache — inner hit only once.
        assert_eq!(inner.lookups.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn store_writes_through_and_warms_cache() {
        let inner = Arc::new(Counting::default());
        let lru = LruFileResolver::new(inner.clone(), 8);

        lru.store("a", &scope(), handle("file-stored", None))
            .await
            .unwrap();
        assert_eq!(inner.stores.load(Ordering::SeqCst), 1, "wrote through");

        // Subsequent lookup is served from the cache the store warmed.
        let got = lru.lookup("a", &scope()).await.unwrap().unwrap();
        assert_eq!(got.uri, "file-stored");
        assert_eq!(inner.lookups.load(Ordering::SeqCst), 0, "no inner lookup");
    }

    #[tokio::test]
    async fn expired_cache_entry_is_evicted_and_refetched() {
        let inner = Arc::new(Counting::default());
        let lru = LruFileResolver::new(inner.clone(), 8);

        // Warm the cache via a first lookup that returns an already-expired
        // handle (epoch is far in the past).
        *inner.handle.lock().unwrap() = Some(handle("file-stale", Some(0)));
        let first = lru.lookup("a", &scope()).await.unwrap().unwrap();
        assert_eq!(first.uri, "file-stale");
        assert_eq!(inner.lookups.load(Ordering::SeqCst), 1);

        // Inner now has a fresh handle; the stale cache entry must be evicted
        // and the inner resolver consulted again.
        *inner.handle.lock().unwrap() = Some(handle("file-fresh", None));
        let got = lru.lookup("a", &scope()).await.unwrap().unwrap();
        assert_eq!(got.uri, "file-fresh");
        assert_eq!(inner.lookups.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn capacity_evicts_least_recently_used() {
        let inner = Arc::new(Counting::default());
        let lru = LruFileResolver::new(inner.clone(), 2);
        lru.store("a", &scope(), handle("A", None)).await.unwrap();
        lru.store("b", &scope(), handle("B", None)).await.unwrap();
        // Touch "a" so "b" becomes the LRU.
        let _ = lru.lookup("a", &scope()).await.unwrap();
        lru.store("c", &scope(), handle("C", None)).await.unwrap();

        // "b" should have been evicted; looking it up now forwards to inner
        // (which returns None since only the latest store is held).
        *inner.handle.lock().unwrap() = None;
        let b = lru.lookup("b", &scope()).await.unwrap();
        assert!(b.is_none());
        assert!(inner.lookups.load(Ordering::SeqCst) >= 1, "b forwarded");
        // "a" still cached (no inner lookup needed).
        let before = inner.lookups.load(Ordering::SeqCst);
        let a = lru.lookup("a", &scope()).await.unwrap().unwrap();
        assert_eq!(a.uri, "A");
        assert_eq!(inner.lookups.load(Ordering::SeqCst), before, "a from cache");
    }
}
