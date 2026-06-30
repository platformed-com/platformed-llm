//! Shared "resolve pass": turn caller-opaque file [`Ref`](crate::FileSource::Ref)s
//! into wire-ready references, uploading to the provider on a registry miss.
//!
//! Runs as an async pre-pass before each provider's *sync* `convert_request`,
//! so the wire-building code stays synchronous and only ever sees a resolved
//! `(id -> ResolvedRef)` map.
//!
//! Per-provider reality (see each provider's [`ProviderUploader`] impl):
//! - **OpenAI** uploads a streamed file to `POST /v1/files` and references it
//!   by `file_id`.
//! - **Gemini (Vertex)** uploads a streamed file to a configured Cloud Storage
//!   bucket (reusing the Vertex OAuth token) and references the resulting
//!   `gs://` URI. With no bucket configured, a `Stream` is rejected and the
//!   resolver must return a durable `gs://` URI via
//!   [`ResolvedFile::ProviderHandle`].
//! - **Anthropic (Vertex)** has no library-owned file store, so a `Stream` is
//!   rejected — the resolver must return a durable handle (or, for a public
//!   file, a public URL via [`ResolvedFile::Url`](crate::ResolvedFile::Url)).

use std::collections::{HashMap, HashSet};
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;

use crate::types::files::now_unix_secs;
use crate::types::{
    FileResolver, FileSource, InputItem, ProviderScope, ResolvedFile, ResolvedHandle, UserPart,
};
use crate::Error;

/// Refresh a handle this many seconds before its stated expiry, so an upload
/// that is valid at resolve time doesn't lapse mid-request.
const EXPIRY_MARGIN_SECS: i64 = 60;

/// A `Ref` resolved to something a provider's `convert_request` can emit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResolvedRef {
    /// Reference a provider-native handle (file id, `gs://` URI, …).
    Handle {
        /// The provider reference.
        uri: String,
        /// MIME type of the referenced file.
        media_type: String,
    },
    /// Reference a plain URL the provider fetches itself.
    Url {
        /// The URL.
        uri: String,
        /// MIME type of the referenced file.
        media_type: String,
    },
}

/// Provider-specific upload of a streamed file. Implemented by providers that
/// own a file store (OpenAI); Vertex providers return an error directing the
/// caller to supply a durable handle/URL instead.
#[async_trait]
pub(crate) trait ProviderUploader: Sync {
    /// Upload `body` and return a provider handle referencing it.
    async fn upload(
        &self,
        media_type: &str,
        content_length: Option<u64>,
        body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>,
    ) -> Result<ResolvedHandle, Error>;
}

fn handle_is_fresh(h: &ResolvedHandle, now_secs: i64) -> bool {
    match h.expires_at {
        None => true,
        Some(exp) => now_secs + EXPIRY_MARGIN_SECS < exp,
    }
}

/// Best-effort file extension (no leading dot) for a MIME type — used to give
/// an uploaded object a friendly name. Empty for unknown types. Shared by the
/// OpenAI (`/v1/files` filename) and Gemini (GCS object name) uploaders.
#[cfg(any(feature = "openai", feature = "google"))]
pub(crate) fn media_type_extension(media_type: &str) -> &'static str {
    match media_type {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "application/pdf" => "pdf",
        "text/plain" => "txt",
        "video/mp4" => "mp4",
        "video/quicktime" | "video/mov" => "mov",
        "video/webm" => "webm",
        _ => "",
    }
}

/// Percent-encode `s` for a URL path/query segment, encoding everything outside
/// the RFC 3986 unreserved set (so `/` becomes `%2F`). Used when building the
/// GCS upload URL's `name=` parameter.
#[cfg(feature = "google")]
pub(crate) fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Walk every user turn (including nested tool-result content) and collect the
/// distinct file-`Ref` ids in first-seen order.
fn collect_ref_ids(items: &[InputItem]) -> Vec<String> {
    fn walk(parts: &[UserPart], ids: &mut Vec<String>, seen: &mut HashSet<String>) {
        for p in parts {
            match p {
                UserPart::Image(FileSource::Ref(id))
                | UserPart::Audio(FileSource::Ref(id))
                | UserPart::Document(FileSource::Ref(id))
                | UserPart::Video(FileSource::Ref(id))
                    if seen.insert(id.clone()) =>
                {
                    ids.push(id.clone());
                }
                UserPart::ToolResult { content, .. } => walk(content, ids, seen),
                _ => {}
            }
        }
    }
    let mut ids = Vec::new();
    let mut seen = HashSet::new();
    for item in items {
        if let InputItem::User { content } = item {
            walk(content, &mut ids, &mut seen);
        }
    }
    ids
}

/// Resolve every file `Ref` in `items` to a wire-ready reference: a fresh
/// [`lookup`](FileResolver::lookup) hit is reused, otherwise the file is
/// [`open`](FileResolver::open)ed and uploaded.
///
/// Errors if a `Ref` is present but no resolver was configured.
pub(crate) async fn resolve_refs(
    items: &[InputItem],
    scope: &ProviderScope,
    resolver: Option<&dyn FileResolver>,
    uploader: &dyn ProviderUploader,
) -> Result<HashMap<String, ResolvedRef>, Error> {
    let ids = collect_ref_ids(items);
    let mut map = HashMap::new();
    if ids.is_empty() {
        return Ok(map);
    }
    let resolver = resolver.ok_or_else(|| {
        Error::config(
            "prompt references a file by id (Ref) but no FileResolver was configured \
             on the provider — call .with_file_resolver(..)",
        )
    })?;

    let now = now_unix_secs();
    for id in ids {
        if let Some(h) = resolver.lookup(&id, scope).await? {
            if handle_is_fresh(&h, now) {
                map.insert(
                    id,
                    ResolvedRef::Handle {
                        uri: h.uri,
                        media_type: h.media_type,
                    },
                );
                continue;
            }
        }
        match resolver.open(&id, scope).await? {
            ResolvedFile::Stream {
                media_type,
                content_length,
                body,
            } => {
                let handle = uploader.upload(&media_type, content_length, body).await?;
                resolver.store(&id, scope, handle.clone()).await?;
                map.insert(
                    id,
                    ResolvedRef::Handle {
                        uri: handle.uri,
                        media_type: handle.media_type,
                    },
                );
            }
            ResolvedFile::ProviderHandle(h) => {
                map.insert(
                    id,
                    ResolvedRef::Handle {
                        uri: h.uri,
                        media_type: h.media_type,
                    },
                );
            }
            ResolvedFile::Url { uri, media_type } => {
                map.insert(id, ResolvedRef::Url { uri, media_type });
            }
        }
    }
    Ok(map)
}

/// Uploader for providers with no library-owned file store (Vertex Gemini
/// without a bucket / Anthropic). Rejects `Stream` payloads, pointing the
/// caller at the durable handle/URL path.
#[cfg(any(feature = "google", feature = "anthropic-vertex"))]
pub(crate) struct NoLibraryUpload {
    /// Provider label for the error message.
    pub provider: &'static str,
}

#[cfg(any(feature = "google", feature = "anthropic-vertex"))]
#[async_trait]
impl ProviderUploader for NoLibraryUpload {
    async fn upload(
        &self,
        _media_type: &str,
        _content_length: Option<u64>,
        body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>,
    ) -> Result<ResolvedHandle, Error> {
        drop(body);
        Err(Error::config(format!(
            "{} has no library-owned file store; have your FileResolver return a durable \
             reference (e.g. a gs:// URI) via ResolvedFile::ProviderHandle — or a public URL \
             via ResolvedFile::Url — instead of a Stream",
            self.provider
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::ProviderType;
    use futures_util::stream;
    use std::sync::Mutex;

    /// In-memory fake resolver recording calls, for resolve-pass tests.
    struct FakeResolver {
        /// id -> handle returned by lookup (None = miss).
        cached: Mutex<HashMap<String, ResolvedHandle>>,
        /// id -> what open() returns.
        opens: Mutex<HashMap<String, ResolvedFile>>,
        /// recorded (id) of store() calls.
        stored: Mutex<Vec<(String, ResolvedHandle)>>,
        /// count of open() calls per id.
        open_calls: Mutex<HashMap<String, usize>>,
    }

    impl FakeResolver {
        fn new() -> Self {
            Self {
                cached: Mutex::new(HashMap::new()),
                opens: Mutex::new(HashMap::new()),
                stored: Mutex::new(Vec::new()),
                open_calls: Mutex::new(HashMap::new()),
            }
        }
    }

    #[async_trait]
    impl FileResolver for FakeResolver {
        async fn lookup(
            &self,
            id: &str,
            _scope: &ProviderScope,
        ) -> Result<Option<ResolvedHandle>, Error> {
            Ok(self.cached.lock().unwrap().get(id).cloned())
        }
        async fn open(&self, id: &str, _scope: &ProviderScope) -> Result<ResolvedFile, Error> {
            *self
                .open_calls
                .lock()
                .unwrap()
                .entry(id.to_string())
                .or_insert(0) += 1;
            self.opens
                .lock()
                .unwrap()
                .remove(id)
                .ok_or_else(|| Error::config(format!("no open scripted for {id}")))
        }
        async fn store(
            &self,
            id: &str,
            _scope: &ProviderScope,
            handle: ResolvedHandle,
        ) -> Result<(), Error> {
            self.stored.lock().unwrap().push((id.to_string(), handle));
            Ok(())
        }
    }

    /// Uploader that hands back a deterministic handle and counts calls.
    struct CountingUploader {
        calls: Mutex<usize>,
    }
    #[async_trait]
    impl ProviderUploader for CountingUploader {
        async fn upload(
            &self,
            media_type: &str,
            _content_length: Option<u64>,
            body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>,
        ) -> Result<ResolvedHandle, Error> {
            // Drain so the stream is genuinely consumed.
            use futures_util::StreamExt as _;
            let mut b = body;
            while let Some(c) = b.next().await {
                c?;
            }
            let mut n = self.calls.lock().unwrap();
            *n += 1;
            Ok(ResolvedHandle {
                uri: format!("file-uploaded-{}", *n),
                media_type: media_type.to_string(),
                expires_at: None,
            })
        }
    }

    fn scope() -> ProviderScope {
        ProviderScope::new(ProviderType::OpenAI, "test")
    }

    fn img_ref(id: &str) -> InputItem {
        InputItem::User {
            content: vec![UserPart::Image(FileSource::Ref(id.to_string()))],
        }
    }

    fn stream_file(media_type: &str) -> ResolvedFile {
        ResolvedFile::Stream {
            media_type: media_type.to_string(),
            content_length: Some(3),
            body: Box::pin(stream::once(async { Ok(Bytes::from_static(b"abc")) })),
        }
    }

    #[tokio::test]
    async fn lookup_hit_skips_upload() {
        let r = FakeResolver::new();
        r.cached.lock().unwrap().insert(
            "a".into(),
            ResolvedHandle {
                uri: "file-cached".into(),
                media_type: "application/pdf".into(),
                expires_at: None,
            },
        );
        let up = CountingUploader {
            calls: Mutex::new(0),
        };
        let map = resolve_refs(&[img_ref("a")], &scope(), Some(&r), &up)
            .await
            .unwrap();
        assert_eq!(
            map.get("a"),
            Some(&ResolvedRef::Handle {
                uri: "file-cached".into(),
                media_type: "application/pdf".into()
            })
        );
        assert_eq!(*up.calls.lock().unwrap(), 0, "no upload on cache hit");
    }

    #[tokio::test]
    async fn lookup_miss_uploads_and_stores() {
        let r = FakeResolver::new();
        r.opens
            .lock()
            .unwrap()
            .insert("a".into(), stream_file("application/pdf"));
        let up = CountingUploader {
            calls: Mutex::new(0),
        };
        let map = resolve_refs(&[img_ref("a")], &scope(), Some(&r), &up)
            .await
            .unwrap();
        assert!(
            matches!(map.get("a"), Some(ResolvedRef::Handle { uri, .. }) if uri == "file-uploaded-1")
        );
        assert_eq!(*up.calls.lock().unwrap(), 1);
        assert_eq!(
            r.stored.lock().unwrap().len(),
            1,
            "uploaded handle persisted"
        );
    }

    #[tokio::test]
    async fn provider_handle_used_verbatim_no_store() {
        let r = FakeResolver::new();
        r.opens.lock().unwrap().insert(
            "a".into(),
            ResolvedFile::ProviderHandle(ResolvedHandle {
                uri: "gs://bucket/x.pdf".into(),
                media_type: "application/pdf".into(),
                expires_at: None,
            }),
        );
        let up = CountingUploader {
            calls: Mutex::new(0),
        };
        let map = resolve_refs(&[img_ref("a")], &scope(), Some(&r), &up)
            .await
            .unwrap();
        assert!(
            matches!(map.get("a"), Some(ResolvedRef::Handle { uri, .. }) if uri == "gs://bucket/x.pdf")
        );
        assert_eq!(*up.calls.lock().unwrap(), 0);
        assert!(
            r.stored.lock().unwrap().is_empty(),
            "caller-owned handle not re-stored"
        );
    }

    #[tokio::test]
    async fn expired_cached_handle_forces_reupload() {
        let r = FakeResolver::new();
        r.cached.lock().unwrap().insert(
            "a".into(),
            ResolvedHandle {
                uri: "file-stale".into(),
                media_type: "application/pdf".into(),
                // already expired (epoch is far in the past)
                expires_at: Some(0),
            },
        );
        r.opens
            .lock()
            .unwrap()
            .insert("a".into(), stream_file("application/pdf"));
        let up = CountingUploader {
            calls: Mutex::new(0),
        };
        let map = resolve_refs(&[img_ref("a")], &scope(), Some(&r), &up)
            .await
            .unwrap();
        assert!(
            matches!(map.get("a"), Some(ResolvedRef::Handle { uri, .. }) if uri == "file-uploaded-1")
        );
        assert_eq!(
            *up.calls.lock().unwrap(),
            1,
            "stale handle triggered re-upload"
        );
    }

    #[tokio::test]
    async fn ref_without_resolver_errors() {
        let up = CountingUploader {
            calls: Mutex::new(0),
        };
        let err = resolve_refs(&[img_ref("a")], &scope(), None, &up)
            .await
            .expect_err("Ref needs a resolver");
        assert!(matches!(err, Error::Config(_)));
    }

    #[tokio::test]
    async fn no_refs_returns_empty_without_touching_resolver() {
        let up = CountingUploader {
            calls: Mutex::new(0),
        };
        let plain = InputItem::user("hi");
        let map = resolve_refs(&[plain], &scope(), None, &up).await.unwrap();
        assert!(map.is_empty());
    }
}
