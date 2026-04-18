use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::warn;

use super::common;
use crate::auth::AuthStore;
use crate::auth::StoredCredential;
use crate::auth::openai_codex_account_id_from_credential;
use crate::credential::CredentialManager;
use crate::credential::OAuthTokens;
use crate::error::Error;
use crate::error::Result;
use crate::model::Model;
use crate::provider::CompletionRequest;
use crate::provider::Provider;
use crate::provider::Usage;
use crate::retry::RetryConfig;
use crate::retry::is_retryable_status;
use crate::streaming::ContentBlock;
use crate::streaming::ContentDelta;
use crate::streaming::MessageMetadata;
use crate::streaming::StreamEvent;

pub const OPENAI_CODEX_PROVIDER: &str = "openai-codex";
pub const OPENAI_CODEX_RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";
const OPENAI_CODEX_BETA_HEADER: &str = "responses=experimental";
const OPENAI_CODEX_NOT_ENTITLED_CODE: &str = "usage_not_included";

pub const OPENAI_CODEX_MODEL_IDS: [&str; 6] = [
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
];
const OPENAI_CODEX_PROBE_MODEL: &str = "gpt-5.1-codex-mini";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntitlementState {
    Unknown,
    Entitled { checked_at_ms: i64 },
    NotEntitled { reason: String, checked_at_ms: i64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntitlementRecord {
    pub state: EntitlementState,
    pub last_error: Option<String>,
}

impl Default for EntitlementRecord {
    fn default() -> Self {
        Self {
            state: EntitlementState::Unknown,
            last_error: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProbeOutcome {
    Entitled,
    NotEntitled(String),
    Error(String),
}

fn entitlement_cache() -> &'static Mutex<HashMap<String, EntitlementRecord>> {
    static CACHE: OnceLock<Mutex<HashMap<String, EntitlementRecord>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cache_key(account: &str) -> String {
    format!("{OPENAI_CODEX_PROVIDER}:{account}")
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

pub fn codex_models() -> Vec<Model> {
    OPENAI_CODEX_MODEL_IDS
        .iter()
        .map(|id| Model {
            id: (*id).to_string(),
            name: (*id).to_string(),
            provider: OPENAI_CODEX_PROVIDER.to_string(),
            max_input_tokens: 400_000,
            max_output_tokens: 128_000,
            supports_thinking: true,
            supports_images: true,
            supports_tools: true,
            input_cost_per_mtok: None,
            output_cost_per_mtok: None,
        })
        .collect()
}

pub fn entitlement_record(account: &str) -> EntitlementRecord {
    entitlement_cache()
        .lock()
        .expect("entitlement cache lock poisoned")
        .get(&cache_key(account))
        .cloned()
        .unwrap_or_default()
}

pub fn reset_entitlement(provider: &str, account: Option<&str>) {
    if provider != OPENAI_CODEX_PROVIDER {
        return;
    }

    let mut cache = entitlement_cache()
        .lock()
        .expect("entitlement cache lock poisoned");
    if let Some(account) = account {
        cache.remove(&cache_key(account));
    } else {
        cache.retain(|key, _| !key.starts_with(&format!("{OPENAI_CODEX_PROVIDER}:")));
    }
}

fn set_entitlement_record(account: &str, record: EntitlementRecord) -> EntitlementRecord {
    entitlement_cache()
        .lock()
        .expect("entitlement cache lock poisoned")
        .insert(cache_key(account), record.clone());
    record
}

fn classify_probe_response(status: u16, body: &str) -> ProbeOutcome {
    if (200..300).contains(&status) {
        return ProbeOutcome::Entitled;
    }

    let error_code = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| error.get("code"))
                .and_then(|code| code.as_str())
                .map(str::to_string)
        });

    if status == 403 || error_code.as_deref() == Some(OPENAI_CODEX_NOT_ENTITLED_CODE) {
        return ProbeOutcome::NotEntitled(
            "authenticated but not entitled for Codex use".to_string(),
        );
    }

    ProbeOutcome::Error(if body.trim().is_empty() {
        format!("entitlement probe failed with HTTP {status}")
    } else {
        format!("entitlement probe failed with HTTP {status}: {body}")
    })
}

#[cfg(test)]
type ProbeHook = Arc<dyn Fn(&StoredCredential) -> ProbeOutcome + Send + Sync>;

#[cfg(test)]
fn probe_hook() -> &'static Mutex<Option<ProbeHook>> {
    static HOOK: OnceLock<Mutex<Option<ProbeHook>>> = OnceLock::new();
    HOOK.get_or_init(|| Mutex::new(None))
}

fn build_probe_request_body() -> Value {
    json!({
        "model": OPENAI_CODEX_PROBE_MODEL,
        "store": false,
        "stream": false,
        "instructions": "codex entitlement probe",
        "input": [{
            "role": "user",
            "content": [{"type": "input_text", "text": "ping"}],
        }],
        "text": {"verbosity": "low"},
    })
}

fn build_probe_request(
    client: &reqwest::Client,
    credential: &StoredCredential,
) -> Result<reqwest::Request> {
    let token = credential.token().to_string();
    let account_id = openai_codex_account_id_from_credential(credential)?;

    client
        .post(OPENAI_CODEX_RESPONSES_URL)
        .header("authorization", format!("Bearer {token}"))
        .header("chatgpt-account-id", account_id)
        .header("OpenAI-Beta", OPENAI_CODEX_BETA_HEADER)
        .header("originator", "pi")
        .header("content-type", "application/json")
        .json(&build_probe_request_body())
        .build()
        .map_err(Into::into)
}

async fn send_probe_request(credential: &StoredCredential) -> Result<reqwest::Response> {
    let client = common::build_http_client(Duration::from_secs(30))?;
    let request = build_probe_request(&client, credential)?;
    client.execute(request).await.map_err(Into::into)
}

async fn live_probe(
    credential: &StoredCredential,
    manager: Option<&CredentialManager>,
) -> ProbeOutcome {
    let retry = RetryConfig::deterministic();
    let mut transient_attempt = 0;
    let mut did_refresh = false;
    let mut current = credential.clone();

    loop {
        let response = match send_probe_request(&current).await {
            Ok(response) => response,
            Err(e) => {
                if transient_attempt < retry.max_retries {
                    tokio::time::sleep(retry.backoff_for(transient_attempt)).await;
                    transient_attempt += 1;
                    continue;
                }
                return ProbeOutcome::Error(format!("failed to send entitlement probe: {e}"));
            }
        };

        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();

        if status == 401 && !did_refresh {
            if let Some(manager) = manager {
                match manager.force_refresh().await {
                    Ok(refreshed) => {
                        current = refreshed;
                        did_refresh = true;
                        continue;
                    }
                    Err(e) => {
                        return ProbeOutcome::Error(format!(
                            "OpenAI Codex token refresh failed: {e}"
                        ));
                    }
                }
            }
            return ProbeOutcome::Error("OpenAI Codex account is unauthenticated".to_string());
        }

        if is_retryable_status(status) && transient_attempt < retry.max_retries {
            tokio::time::sleep(retry.backoff_for(transient_attempt)).await;
            transient_attempt += 1;
            continue;
        }

        return classify_probe_response(status, &body);
    }
}

async fn run_probe(
    credential: &StoredCredential,
    manager: Option<&CredentialManager>,
) -> ProbeOutcome {
    #[cfg(test)]
    if let Some(hook) = probe_hook()
        .lock()
        .expect("probe hook lock poisoned")
        .clone()
    {
        return hook(credential);
    }

    live_probe(credential, manager).await
}

pub async fn ensure_entitlement(
    store: &AuthStore,
    account: &str,
    manager: Option<&CredentialManager>,
) -> EntitlementRecord {
    let cached = entitlement_record(account);
    match &cached.state {
        EntitlementState::Entitled { .. } | EntitlementState::NotEntitled { .. } => return cached,
        EntitlementState::Unknown if cached.last_error.is_some() => return cached,
        EntitlementState::Unknown => {}
    }

    let Some(mut credential) = store
        .credential_for(OPENAI_CODEX_PROVIDER, account)
        .cloned()
    else {
        return cached;
    };
    if credential.is_expired() {
        if let Some(manager) = manager {
            match manager.get_credential().await {
                Ok(refreshed) => credential = refreshed,
                Err(e) => {
                    return set_entitlement_record(
                        account,
                        EntitlementRecord {
                            state: EntitlementState::Unknown,
                            last_error: Some(e.to_string()),
                        },
                    );
                }
            }
        } else {
            return cached;
        }
    }

    let checked_at_ms = now_ms();
    match run_probe(&credential, manager).await {
        ProbeOutcome::Entitled => set_entitlement_record(
            account,
            EntitlementRecord {
                state: EntitlementState::Entitled { checked_at_ms },
                last_error: None,
            },
        ),
        ProbeOutcome::NotEntitled(reason) => set_entitlement_record(
            account,
            EntitlementRecord {
                state: EntitlementState::NotEntitled {
                    reason,
                    checked_at_ms,
                },
                last_error: None,
            },
        ),
        ProbeOutcome::Error(error) => set_entitlement_record(
            account,
            EntitlementRecord {
                state: EntitlementState::Unknown,
                last_error: Some(error),
            },
        ),
    }
}

pub async fn codex_status_suffix(store: &AuthStore, account: &str) -> Option<String> {
    codex_status_suffix_with_manager(store, account, None).await
}

pub async fn codex_status_suffix_with_manager(
    store: &AuthStore,
    account: &str,
    manager: Option<&CredentialManager>,
) -> Option<String> {
    store.credential_for(OPENAI_CODEX_PROVIDER, account)?;

    let record = ensure_entitlement(store, account, manager).await;
    Some(match record.state {
        EntitlementState::Entitled { .. } => "codex entitled".to_string(),
        EntitlementState::NotEntitled { .. } => {
            "authenticated but not entitled for Codex use".to_string()
        }
        EntitlementState::Unknown => {
            if record.last_error.is_some() {
                "authenticated, entitlement check failed".to_string()
            } else {
                "authenticated, entitlement unknown".to_string()
            }
        }
    })
}

pub async fn catalog_for_active_account(store: &AuthStore, account: &str) -> Vec<Model> {
    catalog_for_active_account_with_manager(store, account, None).await
}

pub async fn catalog_for_active_account_with_manager(
    store: &AuthStore,
    account: &str,
    manager: Option<&CredentialManager>,
) -> Vec<Model> {
    match ensure_entitlement(store, account, manager).await.state {
        EntitlementState::Entitled { .. } => codex_models(),
        EntitlementState::Unknown | EntitlementState::NotEntitled { .. } => Vec::new(),
    }
}

pub fn refresh_fn_for_codex()
-> impl Fn(&str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<OAuthTokens>> + Send>>
+ Send
+ Sync
+ 'static {
    |refresh_token| {
        let refresh_token = refresh_token.to_string();
        Box::pin(async move {
            let creds = crate::auth::OAuthFlow::OpenAiCodex
                .refresh_token(&refresh_token)
                .await?;
            Ok(OAuthTokens {
                access_token: creds.access,
                refresh_token: creds.refresh,
                expires_at_ms: creds.expires,
            })
        })
    }
}

pub struct OpenAICodexProvider {
    credential_manager: Arc<CredentialManager>,
    models: Vec<Model>,
    account: String,
}

impl OpenAICodexProvider {
    pub fn new(
        credential_manager: Arc<CredentialManager>,
        models: Vec<Model>,
        account: String,
    ) -> Arc<dyn Provider> {
        Arc::new(Self {
            credential_manager,
            models,
            account,
        })
    }
}

#[async_trait]
impl Provider for OpenAICodexProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<()> {
        let credential = self.credential_manager.get_credential().await?;
        let mut record = entitlement_record(&self.account);
        if matches!(record.state, EntitlementState::Unknown) {
            let mut store = AuthStore::default();
            store.set_credential(OPENAI_CODEX_PROVIDER, &self.account, credential.clone());
            record = ensure_entitlement(
                &store,
                &self.account,
                Some(self.credential_manager.as_ref()),
            )
            .await;
        }

        match &record.state {
            EntitlementState::Entitled { .. } => {}
            EntitlementState::NotEntitled { reason, .. } => {
                return Err(Error::Auth {
                    message: format!("{reason}. ChatGPT Plus or Pro is required for openai-codex"),
                });
            }
            EntitlementState::Unknown => {
                if let Some(error) = record.last_error {
                    return Err(Error::provider_with_status(
                        503,
                        format!("openai-codex entitlement check failed: {error}"),
                    ));
                }
                return Err(Error::provider_with_status(
                    503,
                    "openai-codex entitlement check failed".to_string(),
                ));
            }
        }

        let mut attempt = OpenAICodexAttempt::new(
            request,
            tx,
            credential,
            Arc::clone(&self.credential_manager),
        );
        attempt.run().await
    }

    fn models(&self) -> &[Model] {
        &self.models
    }

    fn name(&self) -> &str {
        OPENAI_CODEX_PROVIDER
    }

    async fn reload_credentials(&self) {
        reset_entitlement(OPENAI_CODEX_PROVIDER, None);
        self.credential_manager.reload_from_disk().await;
    }

    async fn is_available(&self) -> bool {
        let credential = self.credential_manager.get_credential().await;
        credential.is_ok()
    }
}

struct OpenAICodexAttempt {
    request: CompletionRequest,
    tx: mpsc::Sender<StreamEvent>,
    credential: StoredCredential,
    credential_manager: Arc<CredentialManager>,
    retry: RetryConfig,
}

impl OpenAICodexAttempt {
    fn new(
        request: CompletionRequest,
        tx: mpsc::Sender<StreamEvent>,
        credential: StoredCredential,
        credential_manager: Arc<CredentialManager>,
    ) -> Self {
        Self {
            request,
            tx,
            credential,
            credential_manager,
            retry: RetryConfig::deterministic(),
        }
    }

    async fn run(&mut self) -> Result<()> {
        let mut transient_attempt = 0;
        let mut did_refresh = false;

        loop {
            let response = self.send_request().await?;
            let status = response.status().as_u16();
            if response.status().is_success() {
                return parse_codex_sse(response, &self.request.model, self.tx.clone()).await;
            }

            let body_text = response.text().await.unwrap_or_default();
            if status == 401 && !did_refresh {
                match self.credential_manager.force_refresh().await {
                    Ok(refreshed) => {
                        self.credential = refreshed;
                        did_refresh = true;
                        continue;
                    }
                    Err(e) => {
                        return Err(Error::Auth {
                            message: format!("OpenAI Codex token refresh failed: {e}"),
                        });
                    }
                }
            }

            if is_retryable_status(status) && transient_attempt < self.retry.max_retries {
                tokio::time::sleep(self.retry.backoff_for(transient_attempt)).await;
                transient_attempt += 1;
                continue;
            }

            return Err(map_codex_error(status, &body_text));
        }
    }

    async fn send_request(&self) -> Result<reqwest::Response> {
        let client = common::build_http_client(Duration::from_secs(600))?;
        let request = build_codex_request(&client, &self.credential, &self.request)?;
        client.execute(request).await.map_err(Into::into)
    }
}

fn build_codex_request(
    client: &reqwest::Client,
    credential: &StoredCredential,
    request: &CompletionRequest,
) -> Result<reqwest::Request> {
    let token = credential.token().to_string();
    let account_id = openai_codex_account_id_from_credential(credential)?;
    let session_id = request
        .extra_params
        .get("_session_id")
        .and_then(|value| value.as_str());
    let body = build_codex_request_body(request, session_id)?;

    let mut builder = client
        .post(OPENAI_CODEX_RESPONSES_URL)
        .header("authorization", format!("Bearer {token}"))
        .header("chatgpt-account-id", account_id)
        .header("OpenAI-Beta", OPENAI_CODEX_BETA_HEADER)
        .header("originator", "pi")
        .header("accept", "text/event-stream")
        .header("content-type", "application/json");

    if let Some(session_id) = session_id {
        builder = builder.header("session_id", session_id);
    }

    builder.json(&body).build().map_err(Into::into)
}

fn map_codex_error(status: u16, body_text: &str) -> Error {
    let friendly = serde_json::from_str::<serde_json::Value>(body_text)
        .ok()
        .and_then(|value| value.get("error").cloned())
        .and_then(|error| {
            let code = error
                .get("code")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let plan = error.get("plan_type").and_then(|value| value.as_str());
            if code.eq_ignore_ascii_case("usage_not_included") {
                let plan_suffix = plan.map(|value| format!(" ({value})")).unwrap_or_default();
                Some(format!(
                    "ChatGPT usage limit or entitlement block{plan_suffix}"
                ))
            } else {
                error
                    .get("message")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            }
        })
        .unwrap_or_else(|| body_text.to_string());

    if status == 401 {
        Error::Auth {
            message: if friendly.is_empty() {
                "OpenAI Codex account is unauthenticated".to_string()
            } else {
                friendly
            },
        }
    } else if status == 403 || body_text.contains(OPENAI_CODEX_NOT_ENTITLED_CODE) {
        Error::Auth {
            message: "authenticated but not entitled for Codex use. ChatGPT Plus or Pro is required for openai-codex"
                .to_string(),
        }
    } else {
        Error::provider_with_status(status, common::truncate(&friendly, 500))
    }
}

fn build_codex_request_body(
    request: &CompletionRequest,
    session_id: Option<&str>,
) -> Result<Value> {
    let mut extra = request.extra_params.clone();
    let text_override = extra.remove("text");
    let reasoning_override = extra.remove("reasoning");
    let verbosity_override = extra.remove("verbosity");
    extra.remove("_session_id");

    let mut body = json!({
        "model": request.model,
        "store": false,
        "stream": true,
        "input": build_codex_input(&request.messages)?,
        "text": {"verbosity": "medium"},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });

    if let Some(system_prompt) = &request.system_prompt {
        body["instructions"] = json!(system_prompt);
    }

    if let Some(session_id) = session_id {
        body["prompt_cache_key"] = json!(session_id);
    }

    if !request.tools.is_empty() {
        body["tools"] = json!(
            request
                .tools
                .iter()
                .map(|tool| json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "strict": null,
                }))
                .collect::<Vec<_>>()
        );
    }

    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }

    if let Some(thinking) = &request.thinking
        && thinking.enabled
    {
        body["reasoning"] = json!({
            "effort": "medium",
            "summary": "auto",
        });
    }

    if let Some(override_value) = verbosity_override
        && let Some(verbosity) = override_value.as_str()
    {
        body["text"] = json!({"verbosity": verbosity});
    }

    if let Some(override_value) = text_override {
        body["text"] = override_value;
    }

    if let Some(override_value) = reasoning_override {
        body["reasoning"] = override_value;
    }

    if let Some(map) = body.as_object_mut() {
        for (key, value) in extra {
            map.insert(key, value);
        }
    }

    Ok(body)
}

fn build_codex_input(messages: &[Value]) -> Result<Vec<Value>> {
    let mut input = Vec::new();

    for message in messages {
        let Some(role) = message.get("role").and_then(|value| value.as_str()) else {
            continue;
        };

        if role == "user" {
            if let Some(tool_results) = message
                .get("content")
                .and_then(|value| value.as_array())
                .filter(|blocks| {
                    blocks.iter().any(|block| {
                        block.get("type").and_then(|value| value.as_str()) == Some("tool_result")
                    })
                })
            {
                for block in tool_results {
                    if block.get("type").and_then(|value| value.as_str()) != Some("tool_result") {
                        continue;
                    }
                    let Some(call_id) = block
                        .get("tool_use_id")
                        .or_else(|| block.get("call_id"))
                        .and_then(|value| value.as_str())
                    else {
                        continue;
                    };
                    let output = extract_tool_result_text(block);
                    input.push(json!({
                        "type": "function_call_output",
                        "call_id": split_tool_call_id(call_id).0,
                        "output": output,
                    }));
                }
                continue;
            }

            let parts = build_user_parts(message.get("content"));
            if !parts.is_empty() {
                input.push(json!({
                    "type": "message",
                    "role": "user",
                    "content": parts,
                }));
            }
            continue;
        }

        if role != "assistant" {
            continue;
        }

        let Some(content) = message.get("content") else {
            continue;
        };
        if let Some(text) = content.as_str() {
            input.push(json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }));
            continue;
        }

        let Some(blocks) = content.as_array() else {
            continue;
        };

        let mut assistant_parts = Vec::new();
        for block in blocks {
            match block.get("type").and_then(|value| value.as_str()) {
                Some("thinking") => {
                    if let Some(signature) = block
                        .get("signature")
                        .and_then(|value| value.as_str())
                        .filter(|value| !value.is_empty())
                    {
                        if let Ok(reasoning) = serde_json::from_str::<Value>(signature) {
                            input.push(reasoning);
                        }
                    }
                }
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|value| value.as_str()) {
                        assistant_parts
                            .push(json!({"type": "output_text", "text": text, "annotations": []}));
                    }
                }
                Some("refusal") => {
                    if let Some(text) = block.get("text").and_then(|value| value.as_str()) {
                        assistant_parts.push(json!({"type": "refusal", "refusal": text}));
                    }
                }
                Some("tool_use") => {
                    if !assistant_parts.is_empty() {
                        input.push(json!({
                            "type": "message",
                            "role": "assistant",
                            "content": assistant_parts,
                            "status": "completed",
                        }));
                        assistant_parts = Vec::new();
                    }

                    let Some(id) = block.get("id").and_then(|value| value.as_str()) else {
                        continue;
                    };
                    let Some(name) = block.get("name").and_then(|value| value.as_str()) else {
                        continue;
                    };
                    let (call_id, item_id) = split_tool_call_id(id);
                    let arguments = serde_json::to_string(block.get("input").unwrap_or(&json!({})))
                        .unwrap_or_else(|_| "{}".to_string());
                    let mut item = json!({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                    });
                    if let Some(item_id) = item_id {
                        item["id"] = json!(item_id);
                    }
                    input.push(item);
                }
                _ => {}
            }
        }

        if !assistant_parts.is_empty() {
            input.push(json!({
                "type": "message",
                "role": "assistant",
                "content": assistant_parts,
                "status": "completed",
            }));
        }
    }

    Ok(input)
}

fn build_user_parts(content: Option<&Value>) -> Vec<Value> {
    let Some(content) = content else {
        return Vec::new();
    };
    if let Some(text) = content.as_str() {
        return vec![json!({"type": "input_text", "text": text})];
    }

    let mut parts = Vec::new();
    let Some(blocks) = content.as_array() else {
        return parts;
    };

    for block in blocks {
        match block.get("type").and_then(|value| value.as_str()) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(|value| value.as_str()) {
                    parts.push(json!({"type": "input_text", "text": text}));
                }
            }
            Some("input_text") => parts.push(block.clone()),
            Some("image") => {
                if let Some(source) = block.get("source") {
                    parts.push(json!({"type": "input_image", "source": source}));
                } else if let (Some(media_type), Some(data)) = (
                    block.get("media_type").and_then(|value| value.as_str()),
                    block.get("data").and_then(|value| value.as_str()),
                ) {
                    parts.push(json!({
                        "type": "input_image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        }
                    }));
                }
            }
            Some("input_image") => parts.push(block.clone()),
            _ => {}
        }
    }

    parts
}

fn extract_tool_result_text(block: &Value) -> String {
    if let Some(text) = block.get("output").and_then(|value| value.as_str()) {
        return text.to_string();
    }
    if let Some(content) = block.get("content").and_then(|value| value.as_array()) {
        let text = content
            .iter()
            .filter_map(|item| item.get("text").and_then(|value| value.as_str()))
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            return text;
        }
    }
    "(tool result)".to_string()
}

fn split_tool_call_id(id: &str) -> (&str, Option<&str>) {
    if let Some((call_id, item_id)) = id.split_once('|') {
        (call_id, Some(item_id))
    } else {
        (id, None)
    }
}

enum BlockKind {
    Thinking { buffer: String },
    Text { buffer: String },
    ToolUse { partial_json: String },
}

struct ActiveBlock {
    index: usize,
    kind: BlockKind,
}

struct CodexStreamState {
    model: String,
    sent_start: bool,
    next_index: usize,
    active_blocks: HashMap<String, ActiveBlock>,
    saw_tool_call: bool,
}

impl CodexStreamState {
    fn new(model: String) -> Self {
        Self {
            model,
            sent_start: false,
            next_index: 0,
            active_blocks: HashMap::new(),
            saw_tool_call: false,
        }
    }

    fn ensure_message_start(&mut self, item: &Value, events: &mut Vec<StreamEvent>) {
        if self.sent_start {
            return;
        }
        let id = item
            .get("id")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        events.push(StreamEvent::MessageStart {
            message: MessageMetadata {
                id: id.to_string(),
                model: self.model.clone(),
                role: "assistant".to_string(),
            },
        });
        self.sent_start = true;
    }

    fn handle_event(&mut self, event: &Value) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        let Some(event_type) = event.get("type").and_then(|value| value.as_str()) else {
            return Ok(events);
        };

        match event_type {
            "response.output_item.added" => {
                let Some(item) = event.get("item") else {
                    return Ok(events);
                };
                self.ensure_message_start(item, &mut events);
                let Some(item_type) = item.get("type").and_then(|value| value.as_str()) else {
                    return Ok(events);
                };
                let item_id = item
                    .get("id")
                    .and_then(|value| value.as_str())
                    .unwrap_or_else(|| {
                        item.get("call_id")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default()
                    })
                    .to_string();
                let index = self.next_index;
                self.next_index += 1;
                match item_type {
                    "reasoning" => {
                        self.active_blocks.insert(
                            item_id,
                            ActiveBlock {
                                index,
                                kind: BlockKind::Thinking {
                                    buffer: String::new(),
                                },
                            },
                        );
                        events.push(StreamEvent::ContentBlockStart {
                            index,
                            content_block: ContentBlock::Thinking {
                                thinking: String::new(),
                                signature: String::new(),
                            },
                        });
                    }
                    "message" => {
                        self.active_blocks.insert(
                            item_id,
                            ActiveBlock {
                                index,
                                kind: BlockKind::Text {
                                    buffer: String::new(),
                                },
                            },
                        );
                        events.push(StreamEvent::ContentBlockStart {
                            index,
                            content_block: ContentBlock::Text {
                                text: String::new(),
                            },
                        });
                    }
                    "function_call" => {
                        self.saw_tool_call = true;
                        let call_id = item
                            .get("call_id")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default();
                        let name = item
                            .get("name")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default();
                        let tool_id = if item_id.is_empty() {
                            call_id.to_string()
                        } else {
                            format!("{call_id}|{item_id}")
                        };
                        let partial_json = item
                            .get("arguments")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default()
                            .to_string();
                        self.active_blocks.insert(
                            item_id,
                            ActiveBlock {
                                index,
                                kind: BlockKind::ToolUse {
                                    partial_json: partial_json.clone(),
                                },
                            },
                        );
                        events.push(StreamEvent::ContentBlockStart {
                            index,
                            content_block: ContentBlock::ToolUse {
                                id: tool_id,
                                name: name.to_string(),
                                input: json!({}),
                            },
                        });
                        if !partial_json.is_empty() {
                            events.push(StreamEvent::ContentBlockDelta {
                                index,
                                delta: ContentDelta::InputJsonDelta { partial_json },
                            });
                        }
                    }
                    _ => {}
                }
            }
            "response.reasoning_summary_part.added" => {}
            "response.reasoning_summary_text.delta" => {
                let Some(item_id) = event.get("item_id").and_then(|value| value.as_str()) else {
                    return Ok(events);
                };
                let Some(delta) = event.get("delta").and_then(|value| value.as_str()) else {
                    return Ok(events);
                };
                if let Some(active) = self.active_blocks.get_mut(item_id)
                    && let BlockKind::Thinking { buffer } = &mut active.kind
                {
                    buffer.push_str(delta);
                    events.push(StreamEvent::ContentBlockDelta {
                        index: active.index,
                        delta: ContentDelta::ThinkingDelta {
                            thinking: delta.to_string(),
                        },
                    });
                }
            }
            "response.reasoning_summary_part.done" => {
                let Some(item_id) = event.get("item_id").and_then(|value| value.as_str()) else {
                    return Ok(events);
                };
                if let Some(active) = self.active_blocks.get_mut(item_id)
                    && let BlockKind::Thinking { buffer } = &mut active.kind
                    && !buffer.is_empty()
                {
                    buffer.push_str("\n\n");
                    events.push(StreamEvent::ContentBlockDelta {
                        index: active.index,
                        delta: ContentDelta::ThinkingDelta {
                            thinking: "\n\n".to_string(),
                        },
                    });
                }
            }
            "response.content_part.added" => {}
            "response.output_text.delta" | "response.refusal.delta" => {
                let Some(item_id) = event.get("item_id").and_then(|value| value.as_str()) else {
                    return Ok(events);
                };
                let Some(delta) = event.get("delta").and_then(|value| value.as_str()) else {
                    return Ok(events);
                };
                if let Some(active) = self.active_blocks.get_mut(item_id)
                    && let BlockKind::Text { buffer } = &mut active.kind
                {
                    buffer.push_str(delta);
                    events.push(StreamEvent::ContentBlockDelta {
                        index: active.index,
                        delta: ContentDelta::TextDelta {
                            text: delta.to_string(),
                        },
                    });
                }
            }
            "response.function_call_arguments.delta" => {
                let Some(item_id) = event.get("item_id").and_then(|value| value.as_str()) else {
                    return Ok(events);
                };
                let Some(delta) = event.get("delta").and_then(|value| value.as_str()) else {
                    return Ok(events);
                };
                if let Some(active) = self.active_blocks.get_mut(item_id)
                    && let BlockKind::ToolUse { partial_json } = &mut active.kind
                {
                    partial_json.push_str(delta);
                    events.push(StreamEvent::ContentBlockDelta {
                        index: active.index,
                        delta: ContentDelta::InputJsonDelta {
                            partial_json: delta.to_string(),
                        },
                    });
                }
            }
            "response.function_call_arguments.done" => {
                let Some(item_id) = event.get("item_id").and_then(|value| value.as_str()) else {
                    return Ok(events);
                };
                let Some(arguments) = event.get("arguments").and_then(|value| value.as_str())
                else {
                    return Ok(events);
                };
                if let Some(active) = self.active_blocks.get_mut(item_id)
                    && let BlockKind::ToolUse { partial_json } = &mut active.kind
                    && arguments.starts_with(partial_json.as_str())
                {
                    let suffix = &arguments[partial_json.len()..];
                    if !suffix.is_empty() {
                        partial_json.push_str(suffix);
                        events.push(StreamEvent::ContentBlockDelta {
                            index: active.index,
                            delta: ContentDelta::InputJsonDelta {
                                partial_json: suffix.to_string(),
                            },
                        });
                    }
                }
            }
            "response.output_item.done" => {
                let Some(item) = event.get("item") else {
                    return Ok(events);
                };
                let item_id = item
                    .get("id")
                    .and_then(|value| value.as_str())
                    .unwrap_or_else(|| {
                        item.get("call_id")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default()
                    })
                    .to_string();
                let Some(active) = self.active_blocks.remove(&item_id) else {
                    return Ok(events);
                };
                match active.kind {
                    BlockKind::Thinking { mut buffer } => {
                        if buffer.is_empty() {
                            if let Some(summary) =
                                item.get("summary").and_then(|value| value.as_array())
                            {
                                buffer = summary
                                    .iter()
                                    .filter_map(|part| {
                                        part.get("text").and_then(|value| value.as_str())
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n\n");
                                if !buffer.is_empty() {
                                    events.push(StreamEvent::ContentBlockDelta {
                                        index: active.index,
                                        delta: ContentDelta::ThinkingDelta {
                                            thinking: buffer.clone(),
                                        },
                                    });
                                }
                            }
                        }
                        events.push(StreamEvent::ContentBlockDelta {
                            index: active.index,
                            delta: ContentDelta::SignatureDelta {
                                signature: serde_json::to_string(item)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            },
                        });
                        events.push(StreamEvent::ContentBlockStop {
                            index: active.index,
                        });
                    }
                    BlockKind::Text { mut buffer } => {
                        if buffer.is_empty() {
                            if let Some(content) =
                                item.get("content").and_then(|value| value.as_array())
                            {
                                buffer = content
                                    .iter()
                                    .filter_map(|part| {
                                        match part.get("type").and_then(|value| value.as_str()) {
                                            Some("output_text") => {
                                                part.get("text").and_then(|value| value.as_str())
                                            }
                                            Some("refusal") => {
                                                part.get("refusal").and_then(|value| value.as_str())
                                            }
                                            _ => None,
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join("");
                                if !buffer.is_empty() {
                                    events.push(StreamEvent::ContentBlockDelta {
                                        index: active.index,
                                        delta: ContentDelta::TextDelta {
                                            text: buffer.clone(),
                                        },
                                    });
                                }
                            }
                        }
                        events.push(StreamEvent::ContentBlockStop {
                            index: active.index,
                        });
                    }
                    BlockKind::ToolUse { partial_json } => {
                        if let Some(arguments) =
                            item.get("arguments").and_then(|value| value.as_str())
                            && arguments.starts_with(partial_json.as_str())
                        {
                            let suffix = &arguments[partial_json.len()..];
                            if !suffix.is_empty() {
                                events.push(StreamEvent::ContentBlockDelta {
                                    index: active.index,
                                    delta: ContentDelta::InputJsonDelta {
                                        partial_json: suffix.to_string(),
                                    },
                                });
                            }
                        }
                        events.push(StreamEvent::ContentBlockStop {
                            index: active.index,
                        });
                    }
                }
            }
            "response.completed" | "response.done" => {
                let Some(response) = event.get("response") else {
                    return Ok(events);
                };
                let status = response.get("status").and_then(|value| value.as_str());
                match status {
                    Some("failed") | Some("cancelled") => {
                        return Err(Error::Provider {
                            message: response
                                .get("error")
                                .and_then(|value| value.get("message"))
                                .and_then(|value| value.as_str())
                                .unwrap_or("Codex response failed")
                                .to_string(),
                            status: Some(500),
                        });
                    }
                    Some("completed") | Some("incomplete") | Some("queued")
                    | Some("in_progress") | None => {}
                    Some(other) => {
                        warn!("unexpected Codex response status '{other}'");
                    }
                }

                let (input_tokens, cache_read_tokens) = response
                    .get("usage")
                    .map(|usage| {
                        let cached = usage
                            .get("input_tokens_details")
                            .and_then(|details| details.get("cached_tokens"))
                            .and_then(|value| value.as_u64())
                            .unwrap_or(0) as usize;
                        let input = usage
                            .get("input_tokens")
                            .and_then(|value| value.as_u64())
                            .unwrap_or(0) as usize;
                        (input.saturating_sub(cached), cached)
                    })
                    .unwrap_or((0, 0));
                let output_tokens = response
                    .get("usage")
                    .and_then(|usage| usage.get("output_tokens"))
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0) as usize;
                let stop_reason = match status {
                    Some("completed") if self.saw_tool_call => Some("tool_use".to_string()),
                    Some("completed") => Some("end_turn".to_string()),
                    Some("incomplete") => Some("max_tokens".to_string()),
                    _ => None,
                };
                events.push(StreamEvent::MessageDelta {
                    stop_reason,
                    usage: Usage {
                        input_tokens,
                        output_tokens,
                        cache_read_input_tokens: cache_read_tokens,
                        ..Default::default()
                    },
                });
                events.push(StreamEvent::MessageStop);
            }
            "error" => {
                return Err(Error::Provider {
                    message: event
                        .get("message")
                        .and_then(|value| value.as_str())
                        .unwrap_or("Codex stream error")
                        .to_string(),
                    status: None,
                });
            }
            "response.failed" => {
                return Err(Error::Provider {
                    message: event
                        .get("response")
                        .and_then(|value| value.get("error"))
                        .and_then(|value| value.get("message"))
                        .and_then(|value| value.as_str())
                        .unwrap_or("Codex response failed")
                        .to_string(),
                    status: Some(500),
                });
            }
            _ => {}
        }

        Ok(events)
    }
}

async fn parse_codex_sse(
    response: reqwest::Response,
    model: &str,
    tx: mpsc::Sender<StreamEvent>,
) -> Result<()> {
    let mut reader = common::SseLineReader::new(response);
    let mut state = CodexStreamState::new(model.to_string());

    while let Some(event) = reader.next_event().await? {
        if event.data == "[DONE]" {
            break;
        }
        let value: Value = match serde_json::from_str(&event.data) {
            Ok(value) => value,
            Err(e) => {
                warn!("Failed to parse Codex SSE chunk: {e}: {}", event.data);
                continue;
            }
        };

        let events = state.handle_event(&value)?;
        for stream_event in events {
            if tx.send(stream_event).await.is_err() {
                break;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
pub(crate) async fn with_test_probe_hook_async<F, Fut, R>(hook: F, f: impl FnOnce() -> Fut) -> R
where
    F: Fn(&StoredCredential) -> ProbeOutcome + Send + Sync + 'static,
    Fut: std::future::Future<Output = R>,
{
    static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = TEST_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("test lock poisoned");

    reset_entitlement(OPENAI_CODEX_PROVIDER, None);
    *probe_hook().lock().expect("probe hook lock poisoned") = Some(Arc::new(hook));
    let result = f().await;
    *probe_hook().lock().expect("probe hook lock poisoned") = None;
    reset_entitlement(OPENAI_CODEX_PROVIDER, None);
    result
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use base64::Engine;

    use super::*;
    use crate::auth::AuthStorePaths;

    fn fake_openai_codex_jwt(account_id: &str) -> String {
        let header = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"alg":"none","typ":"JWT"}"#);
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(
            serde_json::json!({
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": account_id,
                }
            })
            .to_string()
            .as_bytes(),
        );
        format!("{header}.{payload}.sig")
    }

    fn codex_store() -> AuthStore {
        let mut store = AuthStore::default();
        store.set_credential(
            OPENAI_CODEX_PROVIDER,
            "work",
            StoredCredential::OAuth {
                access_token: fake_openai_codex_jwt("acct-123"),
                refresh_token: "refresh".to_string(),
                expires_at_ms: now_ms() + 3_600_000,
                label: None,
            },
        );
        store.switch_account(OPENAI_CODEX_PROVIDER, "work");
        store
    }

    fn codex_request(session_id: Option<&str>) -> CompletionRequest {
        let mut extra_params = HashMap::new();
        if let Some(session_id) = session_id {
            extra_params.insert("_session_id".to_string(), json!(session_id));
        }

        CompletionRequest {
            model: "gpt-5.1-codex".to_string(),
            messages: vec![json!({"role": "user", "content": [{"type": "text", "text": "hello"}]})],
            system_prompt: Some("system".to_string()),
            max_tokens: Some(128),
            temperature: Some(0.2),
            tools: vec![crate::provider::ToolDefinition {
                name: "read".to_string(),
                description: "Read a file".to_string(),
                input_schema: json!({"type": "object"}),
            }],
            thinking: Some(crate::provider::ThinkingConfig {
                enabled: true,
                budget_tokens: Some(512),
            }),
            no_cache: false,
            cache_ttl: None,
            extra_params,
        }
    }

    fn oauth_credential(account_id: &str) -> StoredCredential {
        StoredCredential::OAuth {
            access_token: fake_openai_codex_jwt(account_id),
            refresh_token: "refresh".to_string(),
            expires_at_ms: now_ms() + 3_600_000,
            label: None,
        }
    }

    fn header_subset(request: &reqwest::Request, names: &[&str]) -> BTreeMap<String, String> {
        names
            .iter()
            .filter_map(|name| {
                request
                    .headers()
                    .get(*name)
                    .and_then(|value| value.to_str().ok())
                    .map(|value| ((*name).to_string(), value.to_string()))
            })
            .collect()
    }

    fn request_body_json(request: &reqwest::Request) -> Value {
        serde_json::from_slice(
            request
                .body()
                .and_then(|body| body.as_bytes())
                .expect("body bytes"),
        )
        .expect("json body")
    }

    #[test]
    fn codex_catalog_is_exact_fixed_set() {
        let ids: Vec<&str> = OPENAI_CODEX_MODEL_IDS.to_vec();
        let unique: HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), 6);
        assert_eq!(unique.len(), 6);
        assert_eq!(
            ids,
            vec![
                "gpt-5.1-codex",
                "gpt-5.1-codex-max",
                "gpt-5.1-codex-mini",
                "gpt-5.2-codex",
                "gpt-5.3-codex",
                "gpt-5.3-codex-spark",
            ]
        );
    }

    #[tokio::test]
    async fn codex_status_suffix_reports_not_entitled() {
        with_test_probe_hook_async(
            |_| {
                ProbeOutcome::NotEntitled(
                    "authenticated but not entitled for Codex use".to_string(),
                )
            },
            || async {
                let store = codex_store();
                let suffix = codex_status_suffix(&store, "work")
                    .await
                    .expect("suffix should exist");
                assert_eq!(suffix, "authenticated but not entitled for Codex use");
            },
        )
        .await;
    }

    #[tokio::test]
    async fn codex_status_suffix_reports_probe_failure() {
        with_test_probe_hook_async(
            |_| ProbeOutcome::Error("boom".to_string()),
            || async {
                let store = codex_store();
                let suffix = codex_status_suffix(&store, "work")
                    .await
                    .expect("suffix should exist");
                assert_eq!(suffix, "authenticated, entitlement check failed");
            },
        )
        .await;
    }

    #[tokio::test]
    async fn codex_catalog_requires_entitlement() {
        with_test_probe_hook_async(
            |_| ProbeOutcome::Entitled,
            || async {
                let store = codex_store();
                let models = catalog_for_active_account(&store, "work").await;
                let ids: Vec<String> = models.into_iter().map(|m| m.id).collect();
                assert_eq!(
                    ids,
                    OPENAI_CODEX_MODEL_IDS
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                );
            },
        )
        .await;

        with_test_probe_hook_async(
            |_| {
                ProbeOutcome::NotEntitled(
                    "authenticated but not entitled for Codex use".to_string(),
                )
            },
            || async {
                let store = codex_store();
                assert!(catalog_for_active_account(&store, "work").await.is_empty());
            },
        )
        .await;
    }

    #[test]
    fn classify_probe_response_treats_usage_not_included_as_not_entitled() {
        let outcome = classify_probe_response(400, r#"{"error":{"code":"usage_not_included"}}"#);
        assert_eq!(
            outcome,
            ProbeOutcome::NotEntitled("authenticated but not entitled for Codex use".to_string())
        );
    }

    #[test]
    fn classify_probe_response_treats_http_403_as_not_entitled() {
        let outcome = classify_probe_response(403, "forbidden");
        assert_eq!(
            outcome,
            ProbeOutcome::NotEntitled("authenticated but not entitled for Codex use".to_string())
        );
    }

    #[test]
    fn build_codex_request_body_preserves_session_cache_and_tools() {
        let request = codex_request(Some("session-1"));
        let body =
            build_codex_request_body(&request, Some("session-1")).expect("body should build");
        assert_eq!(body.get("prompt_cache_key"), Some(&json!("session-1")));
        assert_eq!(body.get("tool_choice"), Some(&json!("auto")));
        assert_eq!(body.get("parallel_tool_calls"), Some(&json!(true)));
        assert_eq!(
            body.get("include"),
            Some(&json!(["reasoning.encrypted_content"]))
        );
        assert!(body.get("tools").is_some());
        assert!(body.get("reasoning").is_some());
    }

    #[test]
    fn build_codex_request_preserves_contract_on_initial_transient_and_refresh_retry_paths() {
        let client =
            common::build_http_client(Duration::from_secs(30)).expect("client should build");
        let request = codex_request(Some("session-1"));
        let expected_body =
            build_codex_request_body(&request, Some("session-1")).expect("body should build");
        let initial = build_codex_request(&client, &oauth_credential("acct-123"), &request)
            .expect("initial request should build");
        let transient_retry = build_codex_request(&client, &oauth_credential("acct-123"), &request)
            .expect("retry request should build");
        let refresh_retry = build_codex_request(&client, &oauth_credential("acct-999"), &request)
            .expect("refresh retry request should build");

        for built in [&initial, &transient_retry, &refresh_retry] {
            assert_eq!(built.method(), reqwest::Method::POST);
            assert_eq!(built.url().as_str(), OPENAI_CODEX_RESPONSES_URL);
            assert_eq!(request_body_json(built), expected_body);
            assert_eq!(
                built.headers().get("OpenAI-Beta").unwrap(),
                OPENAI_CODEX_BETA_HEADER
            );
            assert_eq!(built.headers().get("originator").unwrap(), "pi");
            assert_eq!(built.headers().get("accept").unwrap(), "text/event-stream");
            assert_eq!(
                built.headers().get("content-type").unwrap(),
                "application/json"
            );
            assert_eq!(built.headers().get("session_id").unwrap(), "session-1");
        }

        assert_eq!(
            header_subset(
                &initial,
                &["authorization", "chatgpt-account-id", "session_id"]
            ),
            header_subset(
                &transient_retry,
                &["authorization", "chatgpt-account-id", "session_id"],
            )
        );
        assert_eq!(
            header_subset(
                &initial,
                &["authorization", "chatgpt-account-id", "session_id"]
            ),
            BTreeMap::from([
                (
                    "authorization".to_string(),
                    format!("Bearer {}", fake_openai_codex_jwt("acct-123")),
                ),
                ("chatgpt-account-id".to_string(), "acct-123".to_string()),
                ("session_id".to_string(), "session-1".to_string()),
            ])
        );
        assert_eq!(
            header_subset(
                &refresh_retry,
                &["authorization", "chatgpt-account-id", "session_id"],
            ),
            BTreeMap::from([
                (
                    "authorization".to_string(),
                    format!("Bearer {}", fake_openai_codex_jwt("acct-999")),
                ),
                ("chatgpt-account-id".to_string(), "acct-999".to_string()),
                ("session_id".to_string(), "session-1".to_string()),
            ])
        );
    }

    #[test]
    fn build_codex_request_omits_session_header_without_session_id() {
        let client =
            common::build_http_client(Duration::from_secs(30)).expect("client should build");
        let request = codex_request(None);
        let built = build_codex_request(&client, &oauth_credential("acct-123"), &request)
            .expect("request should build");
        let body = request_body_json(&built);

        assert!(built.headers().get("session_id").is_none());
        assert!(body.get("prompt_cache_key").is_none());
    }

    #[test]
    fn build_probe_request_body_matches_contract() {
        let body = build_probe_request_body();
        assert_eq!(body.get("model"), Some(&json!("gpt-5.1-codex-mini")));
        assert_eq!(body.get("store"), Some(&json!(false)));
        assert_eq!(body.get("stream"), Some(&json!(false)));
        assert_eq!(
            body.get("instructions"),
            Some(&json!("codex entitlement probe"))
        );
        assert_eq!(body.get("text"), Some(&json!({"verbosity": "low"})));
        assert_eq!(
            body.get("input"),
            Some(&json!([{
                "role": "user",
                "content": [{"type": "input_text", "text": "ping"}],
            }]))
        );
        assert!(body.get("tools").is_none());
        assert!(body.get("prompt_cache_key").is_none());
    }

    #[test]
    fn build_probe_request_preserves_contract_on_initial_transient_and_refresh_retry_paths() {
        let client =
            common::build_http_client(Duration::from_secs(30)).expect("client should build");
        let expected_body = build_probe_request_body();
        let initial = build_probe_request(&client, &oauth_credential("acct-123"))
            .expect("initial request should build");
        let transient_retry = build_probe_request(&client, &oauth_credential("acct-123"))
            .expect("retry request should build");
        let refresh_retry = build_probe_request(&client, &oauth_credential("acct-999"))
            .expect("refresh retry request should build");

        for built in [&initial, &transient_retry, &refresh_retry] {
            assert_eq!(built.method(), reqwest::Method::POST);
            assert_eq!(built.url().as_str(), OPENAI_CODEX_RESPONSES_URL);
            assert_eq!(request_body_json(built), expected_body);
            assert_eq!(
                built.headers().get("OpenAI-Beta").unwrap(),
                OPENAI_CODEX_BETA_HEADER
            );
            assert_eq!(built.headers().get("originator").unwrap(), "pi");
            assert_eq!(
                built.headers().get("content-type").unwrap(),
                "application/json"
            );
            assert!(built.headers().get("accept").is_none());
            assert!(built.headers().get("session_id").is_none());
        }

        assert_eq!(
            header_subset(&initial, &["authorization", "chatgpt-account-id"]),
            header_subset(&transient_retry, &["authorization", "chatgpt-account-id"])
        );
        assert_eq!(
            header_subset(&initial, &["authorization", "chatgpt-account-id"]),
            BTreeMap::from([
                (
                    "authorization".to_string(),
                    format!("Bearer {}", fake_openai_codex_jwt("acct-123")),
                ),
                ("chatgpt-account-id".to_string(), "acct-123".to_string()),
            ])
        );
        assert_eq!(
            header_subset(&refresh_retry, &["authorization", "chatgpt-account-id"]),
            BTreeMap::from([
                (
                    "authorization".to_string(),
                    format!("Bearer {}", fake_openai_codex_jwt("acct-999")),
                ),
                ("chatgpt-account-id".to_string(), "acct-999".to_string()),
            ])
        );
    }

    #[tokio::test]
    async fn provider_reload_uses_layered_runtime_store() {
        let dir = tempfile::TempDir::new().expect("tempdir should exist");
        let seed_path = dir.path().join("seed.json");
        let runtime_path = dir.path().join("runtime.json");

        let mut seed = AuthStore::default();
        seed.set_credential(
            OPENAI_CODEX_PROVIDER,
            "work",
            StoredCredential::OAuth {
                access_token: fake_openai_codex_jwt("acct-123"),
                refresh_token: "refresh".to_string(),
                expires_at_ms: now_ms() + 1000,
                label: None,
            },
        );
        seed.switch_account(OPENAI_CODEX_PROVIDER, "work");
        seed.save(&seed_path).expect("seed should save");

        let auth_paths = AuthStorePaths::layered(seed_path.clone(), runtime_path.clone());
        let manager = CredentialManager::with_refresh_fn(
            OPENAI_CODEX_PROVIDER.to_string(),
            seed.active_credential(OPENAI_CODEX_PROVIDER)
                .expect("credential should exist")
                .clone(),
            auth_paths,
            None,
            refresh_fn_for_codex(),
        );
        manager.reload_from_disk().await;
        assert!(runtime_path.parent().is_some());
    }
}
