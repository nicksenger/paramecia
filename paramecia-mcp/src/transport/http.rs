//! HTTP transport for MCP.

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use std::collections::HashMap;
use std::time::Duration;

use super::Transport;
use crate::error::{McpError, McpResult};

/// HTTP transport for MCP servers.
pub struct HttpTransport {
    client: reqwest::Client,
    url: String,
}

impl HttpTransport {
    /// Create a new HTTP transport.
    ///
    /// # Arguments
    ///
    /// * `url` - Base URL of the MCP server
    /// * `headers` - Additional headers to include
    /// * `timeout` - Request timeout
    pub fn new(
        url: impl Into<String>,
        headers: Option<HashMap<String, String>>,
        timeout: Duration,
    ) -> McpResult<Self> {
        let mut header_map = HeaderMap::new();
        header_map.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(h) = headers {
            for (key, value) in h {
                if let (Ok(name), Ok(val)) =
                    (HeaderName::try_from(&key), HeaderValue::from_str(&value))
                {
                    header_map.insert(name, val);
                }
            }
        }

        let client = reqwest::Client::builder()
            .default_headers(header_map)
            .timeout(timeout)
            .build()
            .map_err(McpError::HttpError)?;

        Ok(Self {
            client,
            url: url.into(),
        })
    }
}

#[async_trait]
impl Transport for HttpTransport {
    async fn send(&self, message: &str) -> McpResult<String> {
        let response = self
            .client
            .post(&self.url)
            .body(message.to_string())
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(McpError::ServerError {
                code: status.as_u16() as i32,
                message: body,
            });
        }

        Ok(response.text().await?)
    }

    async fn close(&self) -> McpResult<()> {
        // HTTP is stateless, nothing to close
        Ok(())
    }
}
