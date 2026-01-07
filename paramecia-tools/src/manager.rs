//! Tool discovery and management.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

use crate::builtins::{Bash, Grep, ReadFile, SearchReplace, Todo, WriteFile};
use crate::error::{ToolError, ToolResult};
use crate::mcp_tool::McpTool;
use crate::types::{Tool, ToolConfig, ToolInfo, ToolPermission};
use paramecia_mcp::client::McpClient;
use paramecia_mcp::protocol::RemoteTool;

type ToolFactory = Box<dyn Fn(ToolConfig) -> Box<dyn Tool> + Send + Sync>;
type ToolFactories = HashMap<String, ToolFactory>;
type ToolInstances = RwLock<HashMap<String, Arc<ToolInstance>>>;

/// A single tool instance protected for concurrent access.
pub struct ToolInstance {
    tool: Mutex<Option<Box<dyn Tool>>>,
    available: Notify,
}

/// Blocking read guard used for synchronous access.
pub struct ToolReadGuard<'a> {
    _guard: tokio::sync::MutexGuard<'a, Option<Box<dyn Tool>>>,
    tool: NonNull<dyn Tool>,
}

impl<'a> std::ops::Deref for ToolReadGuard<'a> {
    type Target = dyn Tool;

    fn deref(&self) -> &Self::Target {
        // Safety: the pointer is constructed while the guard is held and remains
        // valid for the lifetime of the guard.
        unsafe { self.tool.as_ref() }
    }
}

/// Blocking write guard used for synchronous mutable access.
pub struct ToolWriteGuard<'a> {
    _guard: tokio::sync::MutexGuard<'a, Option<Box<dyn Tool>>>,
    tool: NonNull<dyn Tool>,
}

impl<'a> std::ops::Deref for ToolWriteGuard<'a> {
    type Target = dyn Tool;

    fn deref(&self) -> &Self::Target {
        // Safety: the pointer is constructed while the guard is held and remains
        // valid for the lifetime of the guard.
        unsafe { self.tool.as_ref() }
    }
}

impl<'a> std::ops::DerefMut for ToolWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety: the pointer is constructed while the guard is held and remains
        // valid for the lifetime of the guard.
        unsafe { self.tool.as_mut() }
    }
}

impl ToolInstance {
    fn new(tool: Box<dyn Tool>) -> Self {
        Self {
            tool: Mutex::new(Some(tool)),
            available: Notify::new(),
        }
    }

    fn lock_sync(&self) -> tokio::sync::MutexGuard<'_, Option<Box<dyn Tool>>> {
        if let Ok(guard) = self.tool.try_lock() {
            return guard;
        }

        loop {
            if let Ok(guard) = self.tool.try_lock() {
                return guard;
            }

            let notified = self.available.notified();
            futures::executor::block_on(notified);
        }
    }

    /// Blocking read access for synchronous callers.
    pub fn read(&self) -> ToolReadGuard<'_> {
        loop {
            let guard = self.lock_sync();
            if let Some(tool) = guard.as_ref() {
                let tool_ref = NonNull::from(tool.as_ref());
                return ToolReadGuard {
                    _guard: guard,
                    tool: tool_ref,
                };
            }

            let notified = self.available.notified();
            drop(guard);
            futures::executor::block_on(notified);
        }
    }

    /// Blocking write access for synchronous callers.
    pub fn write(&self) -> ToolWriteGuard<'_> {
        loop {
            let mut guard = self.lock_sync();
            if let Some(tool) = guard.as_mut() {
                let tool_ref = NonNull::from(tool.as_mut());
                return ToolWriteGuard {
                    _guard: guard,
                    tool: tool_ref,
                };
            }

            let notified = self.available.notified();
            drop(guard);
            futures::executor::block_on(notified);
        }
    }

    async fn checkout(&self) -> ToolResult<Box<dyn Tool>> {
        loop {
            let mut guard = self.tool.lock().await;
            if let Some(tool) = guard.take() {
                return Ok(tool);
            }

            let notified = self.available.notified();
            drop(guard);
            notified.await;
        }
    }

    async fn return_tool(&self, tool: Box<dyn Tool>) {
        let mut guard = self.tool.lock().await;
        *guard = Some(tool);
        drop(guard);
        self.available.notify_one();
    }

    pub async fn execute(&self, args: serde_json::Value) -> ToolResult<serde_json::Value> {
        let mut tool = self.checkout().await?;
        let result = tool.execute(args).await;
        self.return_tool(tool).await;
        result
    }

    pub async fn inspect<R, F>(&self, f: F) -> ToolResult<R>
    where
        F: FnOnce(&dyn Tool) -> R,
    {
        loop {
            let guard = self.tool.lock().await;
            if let Some(tool) = guard.as_ref() {
                let result = f(tool.as_ref());
                return Ok(result);
            }

            let notified = self.available.notified();
            drop(guard);
            notified.await;
        }
    }

    pub async fn inspect_mut<R, F>(&self, f: F) -> ToolResult<R>
    where
        F: FnOnce(&mut dyn Tool) -> R,
    {
        loop {
            let mut guard = self.tool.lock().await;
            if let Some(tool) = guard.as_mut() {
                let result = f(tool.as_mut());
                return Ok(result);
            }

            let notified = self.available.notified();
            drop(guard);
            notified.await;
        }
    }

    pub fn blocking_inspect<R, F>(&self, f: F) -> ToolResult<R>
    where
        F: FnOnce(&dyn Tool) -> R,
    {
        loop {
            let guard = self.lock_sync();
            if let Some(tool) = guard.as_ref() {
                let result = f(tool.as_ref());
                return Ok(result);
            }

            let notified = self.available.notified();
            drop(guard);
            futures::executor::block_on(notified);
        }
    }

    pub fn blocking_inspect_mut<R, F>(&self, f: F) -> ToolResult<R>
    where
        F: FnOnce(&mut dyn Tool) -> R,
    {
        loop {
            let mut guard = self.lock_sync();
            if let Some(tool) = guard.as_mut() {
                let result = f(tool.as_mut());
                return Ok(result);
            }

            let notified = self.available.notified();
            drop(guard);
            futures::executor::block_on(notified);
        }
    }
}

/// Manages tool discovery and instantiation.
pub struct ToolManager {
    /// Registered tool factories.
    factories: ToolFactories,
    /// Active tool instances.
    instances: ToolInstances,
    /// Default configurations for tools.
    default_configs: HashMap<String, ToolConfig>,
    /// User-provided configuration overrides.
    config_overrides: HashMap<String, ToolConfig>,
}

impl ToolManager {
    /// Create a new tool manager with default builtin tools.
    #[must_use]
    pub fn new() -> Self {
        let mut manager = Self {
            factories: HashMap::new(),
            instances: RwLock::new(HashMap::new()),
            default_configs: HashMap::new(),
            config_overrides: HashMap::new(),
        };

        // Register builtin tools
        manager.register_builtin::<Bash>();
        manager.register_builtin::<ReadFile>();
        manager.register_builtin::<WriteFile>();
        manager.register_builtin::<SearchReplace>();
        manager.register_builtin::<Grep>();
        manager.register_builtin::<Todo>();

        manager
    }

    /// Create a tool manager with configuration overrides.
    #[must_use]
    pub fn with_configs(configs: HashMap<String, ToolConfig>) -> Self {
        let mut manager = Self::new();
        manager.config_overrides = configs;
        manager
    }

    /// Register a builtin tool type.
    fn register_builtin<T>(&mut self)
    where
        T: Tool + Default + 'static,
    {
        let default_tool = T::default();
        let name = default_tool.name().to_string();
        let default_config = default_tool.config().clone();

        self.default_configs.insert(name.clone(), default_config);
        self.factories.insert(
            name,
            Box::new(|_config| Box::new(T::default()) as Box<dyn Tool>),
        );
    }

    /// Register a custom tool factory.
    pub fn register<F>(&mut self, name: impl Into<String>, factory: F, default_config: ToolConfig)
    where
        F: Fn(ToolConfig) -> Box<dyn Tool> + Send + Sync + 'static,
    {
        let name = name.into();
        self.default_configs.insert(name.clone(), default_config);
        self.factories.insert(name, Box::new(factory));
    }

    /// Get the effective configuration for a tool.
    #[must_use]
    pub fn get_tool_config(&self, name: &str) -> ToolConfig {
        let default = self.default_configs.get(name).cloned().unwrap_or_default();

        if let Some(override_config) = self.config_overrides.get(name) {
            // Merge configs: override takes precedence
            ToolConfig {
                permission: override_config.permission,
                workdir: override_config.workdir.clone().or(default.workdir),
                allowlist: if override_config.allowlist.is_empty() {
                    default.allowlist
                } else {
                    override_config.allowlist.clone()
                },
                denylist: if override_config.denylist.is_empty() {
                    default.denylist
                } else {
                    override_config.denylist.clone()
                },
                extra: {
                    let mut merged = default.extra;
                    for (k, v) in &override_config.extra {
                        merged.insert(k.clone(), v.clone());
                    }
                    merged
                },
            }
        } else {
            default
        }
    }

    /// Get or create a tool instance.
    pub fn get(&self, name: &str) -> ToolResult<Arc<ToolInstance>> {
        // Check if already instantiated
        {
            let instances = self.instances.read();
            if let Some(tool) = instances.get(name) {
                return Ok(Arc::clone(tool));
            }
        }

        // Create new instance
        let factory = self.factories.get(name).ok_or_else(|| {
            ToolError::NotFound(format!(
                "Unknown tool: {name}. Available: {:?}",
                self.available_tools()
            ))
        })?;

        let config = self.get_tool_config(name);
        let tool = factory(config);
        let tool = Arc::new(ToolInstance::new(tool));

        {
            let mut instances = self.instances.write();
            instances.insert(name.to_string(), Arc::clone(&tool));
        }

        Ok(tool)
    }

    /// Get information about all available tools.
    #[must_use]
    pub fn available_tools(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// Get tool info for all available tools.
    pub fn tool_infos(&self) -> Vec<ToolInfo> {
        self.factories
            .keys()
            .filter_map(|name| {
                self.get(name)
                    .ok()
                    .and_then(|tool| tool.blocking_inspect(|tool_ref| tool_ref.info()).ok())
            })
            .collect()
    }

    /// Reset all tool instances.
    pub fn reset_all(&self) {
        let mut instances = self.instances.write();
        for tool in instances.values() {
            let _ = tool.blocking_inspect_mut(|tool| {
                tool.reset();
            });
        }
        instances.clear();
    }

    /// Check if a tool is available.
    #[must_use]
    pub fn has_tool(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// Register MCP tools from a client.
    ///
    /// # Arguments
    ///
    /// * `client` - The MCP client to use for tool execution
    /// * `remote_tools` - List of remote tools to register
    ///
    /// # Returns
    ///
    /// The number of tools successfully registered.
    pub fn register_mcp_tools(
        &mut self,
        client: Arc<McpClient>,
        remote_tools: Vec<RemoteTool>,
    ) -> usize {
        let mut registered_count = 0;

        for remote_tool in remote_tools {
            let tool_name = remote_tool.name.clone();

            // Check if tool already exists
            if self.factories.contains_key(&tool_name) {
                continue; // Skip if tool already registered
            }

            // Create MCP tool factory
            let client_clone = Arc::clone(&client);
            let factory = move |_config: ToolConfig| {
                let tool = McpTool::new(&remote_tool, Arc::clone(&client_clone));
                Box::new(tool) as Box<dyn Tool>
            };

            // Register the factory
            self.factories.insert(tool_name.clone(), Box::new(factory));

            // Add default config
            let default_config = ToolConfig {
                permission: ToolPermission::Ask, // Default to asking for MCP tools
                ..Default::default()
            };
            self.default_configs.insert(tool_name, default_config);

            registered_count += 1;
        }

        registered_count
    }
}

impl Default for ToolManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_tools_registered() {
        let manager = ToolManager::new();
        let tools = manager.available_tools();

        assert!(tools.contains(&"bash".to_string()));
        assert!(tools.contains(&"read_file".to_string()));
        assert!(tools.contains(&"write_file".to_string()));
        assert!(tools.contains(&"search_replace".to_string()));
        assert!(tools.contains(&"grep".to_string()));
        assert!(tools.contains(&"todo".to_string()));
    }

    #[test]
    fn test_get_tool() {
        let manager = ToolManager::new();
        let bash = manager.get("bash");
        assert!(bash.is_ok());
    }

    #[test]
    fn test_unknown_tool() {
        let manager = ToolManager::new();
        let result = manager.get("nonexistent");
        assert!(result.is_err());
    }
}
