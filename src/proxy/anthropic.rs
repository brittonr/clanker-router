//! Native Anthropic Messages API proxy endpoint.
//!
//! Accepts `POST /v1/messages` with Anthropic-format request bodies,
//! routes through the standard `Router.complete()` pipeline (credential
//! rotation, rate limits, fallbacks, usage recording), and streams back
//! native Anthropic SSE events.
//!
//! This lets clankers (which speaks native Anthropic) use the router for
//! load balancing without losing Anthropic-specific features (prompt
//! caching breakpoints, extended thinking signatures, cache token reporting).

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::response::sse::Event as SseEvent;
use axum::response::sse::KeepAlive;
use axum::response::sse::Sse;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;
use tracing::warn;

use super::ProxyState;
use crate::provider::CompletionRequest;
use crate::provider::ThinkingConfig;
use crate::provider::ToolDefinition;
use crate::streaming::ContentBlock;
use crate::streaming::ContentDelta;
use crate::streaming::StreamEvent;

// ── Anthropic request types ─────────────────────────────────────────────

/// Top-level Anthropic Messages API request body.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<Value>,
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub system: Option<Vec<Value>>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub thinking: Option<AnthropicThinking>,
}

/// Anthropic tool definition.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<Value>,
    // cache_control passed through in raw JSON
}

/// Anthropic thinking configuration.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicThinking {
    #[serde(rename = "enabled")]
    Enabled {
        #[serde(default = "default_budget")]
        budget_tokens: usize,
    },
    #[serde(rename = "disabled")]
    Disabled,
}

fn default_budget() -> usize {
    10000
}

// ── Request conversion ──────────────────────────────────────────────────

/// Convert an Anthropic Messages API request into the router's `CompletionRequest`.
///
/// Messages pass through as `Vec<Value>` (the Anthropic backend already
/// expects this format). System blocks are preserved in
/// `extra_params["_anthropic_system"]` for round-trip fidelity, with a
/// plain-text `system_prompt` extracted for non-Anthropic fallback backends.
pub(crate) fn convert_anthropic_request(req: AnthropicRequest) -> CompletionRequest {
    let mut extra_params = std::collections::HashMap::new();

    // Preserve raw system blocks for the Anthropic backend
    let system_prompt = if let Some(ref blocks) = req.system {
        extra_params.insert("_anthropic_system".to_string(), json!(blocks));

        // Extract plain text for non-Anthropic fallback backends
        let text_parts: Vec<&str> = blocks
            .iter()
            .filter_map(|b| {
                if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                    b.get("text").and_then(|t| t.as_str())
                } else {
                    None
                }
            })
            .collect();

        if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join("\n\n"))
        }
    } else {
        None
    };

    let tools: Vec<ToolDefinition> = req
        .tools
        .unwrap_or_default()
        .into_iter()
        .map(|t| ToolDefinition {
            name: t.name,
            description: t.description.unwrap_or_default(),
            input_schema: t.input_schema.unwrap_or(json!({"type": "object"})),
        })
        .collect();

    let thinking = req.thinking.map(|t| match t {
        AnthropicThinking::Enabled { budget_tokens } => ThinkingConfig {
            enabled: true,
            budget_tokens: Some(budget_tokens),
        },
        AnthropicThinking::Disabled => ThinkingConfig {
            enabled: false,
            budget_tokens: None,
        },
    });

    CompletionRequest {
        model: req.model,
        messages: req.messages,
        system_prompt,
        max_tokens: Some(req.max_tokens),
        temperature: req.temperature,
        tools,
        thinking,
        no_cache: false,
        cache_ttl: None,
        extra_params,
    }
}

// ── Anthropic SSE response converter ────────────────────────────────────

/// Converts `StreamEvent` values into Anthropic-format SSE event strings.
///
/// Each call to `convert` returns zero or more `(event_type, json_data)` pairs
/// that become `event: <type>\ndata: <json>\n\n` in the SSE stream.
pub(crate) struct AnthropicSseConverter {
    model: String,
    msg_id: String,
}

impl AnthropicSseConverter {
    pub fn new() -> Self {
        Self {
            model: String::new(),
            msg_id: String::new(),
        }
    }

    /// Convert a `StreamEvent` into Anthropic SSE events.
    /// Returns `(event_type, json_data)` pairs.
    pub fn convert(&mut self, event: StreamEvent) -> Vec<(String, String)> {
        let mut out = Vec::new();

        match event {
            StreamEvent::MessageStart { message } => {
                self.model = message.model.clone();
                self.msg_id = message.id.clone();
                let data = json!({
                    "type": "message_start",
                    "message": {
                        "id": message.id,
                        "type": "message",
                        "role": message.role,
                        "model": message.model,
                        "content": [],
                        "stop_reason": null,
                        "usage": { "input_tokens": 0, "output_tokens": 0 }
                    }
                });
                out.push(("message_start".into(), data.to_string()));
            }

            StreamEvent::ContentBlockStart { index, content_block } => {
                let block_json = match content_block {
                    ContentBlock::Text { text } => json!({
                        "type": "text",
                        "text": text,
                    }),
                    ContentBlock::Thinking { thinking, .. } => json!({
                        "type": "thinking",
                        "thinking": thinking,
                    }),
                    ContentBlock::ToolUse { id, name, .. } => json!({
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": {},
                    }),
                };
                let data = json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": block_json,
                });
                out.push(("content_block_start".into(), data.to_string()));
            }

            StreamEvent::ContentBlockDelta { index, delta } => {
                let delta_json = match delta {
                    ContentDelta::TextDelta { text } => json!({
                        "type": "text_delta",
                        "text": text,
                    }),
                    ContentDelta::ThinkingDelta { thinking } => json!({
                        "type": "thinking_delta",
                        "thinking": thinking,
                    }),
                    ContentDelta::InputJsonDelta { partial_json } => json!({
                        "type": "input_json_delta",
                        "partial_json": partial_json,
                    }),
                    ContentDelta::SignatureDelta { signature } => json!({
                        "type": "signature_delta",
                        "signature": signature,
                    }),
                };
                let data = json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": delta_json,
                });
                out.push(("content_block_delta".into(), data.to_string()));
            }

            StreamEvent::ContentBlockStop { index } => {
                let data = json!({
                    "type": "content_block_stop",
                    "index": index,
                });
                out.push(("content_block_stop".into(), data.to_string()));
            }

            StreamEvent::MessageDelta { stop_reason, usage } => {
                let data = json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                    },
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                        "cache_read_input_tokens": usage.cache_read_input_tokens,
                    }
                });
                out.push(("message_delta".into(), data.to_string()));
            }

            StreamEvent::MessageStop => {
                let data = json!({ "type": "message_stop" });
                out.push(("message_stop".into(), data.to_string()));
            }

            StreamEvent::Error { error } => {
                warn!("Anthropic proxy: stream error: {error}");
                let data = json!({
                    "type": "error",
                    "error": {
                        "type": "server_error",
                        "message": error,
                    }
                });
                out.push(("error".into(), data.to_string()));
            }
        }

        out
    }
}

// ── Error responses (Anthropic format) ──────────────────────────────────

/// Build an Anthropic-format error response.
pub(crate) fn anthropic_error_response(status: StatusCode, error_type: &str, message: &str) -> Response {
    let body = json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        }
    });
    (status, axum::Json(body)).into_response()
}

// ── Auth (reuses existing check, Anthropic error format) ────────────────

fn check_auth_anthropic(state: &ProxyState, headers: &HeaderMap) -> Result<(), Response> {
    if state.allowed_keys.is_empty() {
        return Ok(());
    }

    // Accept both Authorization: Bearer <key> and x-api-key: <key>
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
        .unwrap_or("");

    if token.is_empty() {
        return Err(anthropic_error_response(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "Missing API key. Pass via Authorization: Bearer <key> or x-api-key header",
        ));
    }

    if !state.allowed_keys.iter().any(|k| k == token) {
        return Err(anthropic_error_response(
            StatusCode::FORBIDDEN,
            "authentication_error",
            "Invalid API key",
        ));
    }

    Ok(())
}

// ── Handler ─────────────────────────────────────────────────────────────

/// `POST /v1/messages` — native Anthropic Messages API endpoint.
pub(crate) async fn anthropic_messages(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    axum::Json(body): axum::Json<AnthropicRequest>,
) -> Response {
    if let Err(resp) = check_auth_anthropic(&state, &headers) {
        return resp;
    }

    // Validate: stream must be true
    match body.stream {
        Some(true) => {}
        Some(false) => {
            return anthropic_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "Non-streaming requests are not supported. Set \"stream\": true",
            );
        }
        None => {
            return anthropic_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "The \"stream\" field is required and must be true",
            );
        }
    }

    let request = convert_anthropic_request(body);

    debug!(
        "anthropic proxy: streaming completion for model={} messages={}",
        request.model,
        request.messages.len(),
    );

    let (event_tx, mut event_rx) = mpsc::channel::<StreamEvent>(64);
    let (sse_tx, sse_rx) = mpsc::channel::<Result<SseEvent, Infallible>>(64);

    // Spawn the router completion
    let router_handle = {
        let state = Arc::clone(&state);
        tokio::spawn(async move { state.router.complete(request, event_tx).await })
    };

    // Spawn the SSE converter
    tokio::spawn(async move {
        let mut converter = AnthropicSseConverter::new();

        while let Some(event) = event_rx.recv().await {
            let pairs = converter.convert(event);
            for (event_type, json_data) in pairs {
                let sse = SseEvent::default().event(event_type).data(json_data);
                if sse_tx.send(Ok(sse)).await.is_err() {
                    return;
                }
            }
        }

        // Check for completion errors
        match router_handle.await {
            Ok(Err(e)) => {
                warn!("anthropic proxy: completion error: {e}");
                let err_data = json!({
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": e.to_string(),
                    }
                });
                let sse = SseEvent::default().event("error").data(err_data.to_string());
                let _ = sse_tx.send(Ok(sse)).await;
            }
            Err(e) => {
                warn!("anthropic proxy: completion task panicked: {e}");
            }
            Ok(Ok(())) => {}
        }
    });

    let stream = ReceiverStream::new(sse_rx);
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}
