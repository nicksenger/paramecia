//! Tool name aliasing for model-specific compatibility.
//!
//! Different LLM models may have been trained on specific tool names and schemas.
//! This module provides aliasing to map between paramecia's internal tool names
//! and the names expected by specific models (particularly Qwen models).
//!
//! ## Qwen-Code Tool Names
//!
//! The Qwen3-Coder family was trained with qwen-code's tool definitions:
//!
//! | Paramecia Name | Qwen-Code Name | Notes |
//! |----------------|----------------|-------|
//! | `bash` | `run_shell_command` | Includes `is_background` param |
//! | `read_file` | `read_file` | Uses `absolute_path` not `path` |
//! | `write_file` | `write_file` | Uses `file_path` not `path` |
//! | `search_replace` | `edit` | Different parameter structure |
//! | `grep` | `grep_search` | Same parameters |
//! | `todo` | `todo_write` | Same structure |

use crate::model_prompts::ModelFamily;
use serde_json::{Map, Value, json};
use std::collections::HashMap;

/// Tool alias configuration for a specific model family.
#[derive(Debug, Clone)]
pub struct ToolAliasConfig {
    /// Map from internal name to model-expected name.
    pub name_map: HashMap<String, String>,
    /// Map from model name back to internal name.
    pub reverse_map: HashMap<String, String>,
    /// Parameter transformers for each tool.
    pub param_transforms: HashMap<String, ParamTransform>,
}

/// Parameter transformation rules.
#[derive(Debug, Clone, Default)]
pub struct ParamTransform {
    /// Rename parameters: internal_name -> model_name
    pub renames: HashMap<String, String>,
    /// Add default parameters
    pub defaults: HashMap<String, Value>,
    /// Required parameters to add to schema
    pub additional_required: Vec<String>,
    /// Additional properties to add to schema
    pub additional_properties: HashMap<String, Value>,
}

impl ToolAliasConfig {
    /// Create alias config for Qwen-Coder models.
    #[must_use]
    pub fn qwen_coder() -> Self {
        let mut name_map = HashMap::new();
        let mut reverse_map = HashMap::new();
        let mut param_transforms = HashMap::new();

        // bash -> run_shell_command
        name_map.insert("bash".to_string(), "run_shell_command".to_string());
        reverse_map.insert("run_shell_command".to_string(), "bash".to_string());
        param_transforms.insert("bash".to_string(), Self::bash_transform());

        // grep -> grep_search
        name_map.insert("grep".to_string(), "grep_search".to_string());
        reverse_map.insert("grep_search".to_string(), "grep".to_string());

        // search_replace -> edit
        name_map.insert("search_replace".to_string(), "edit".to_string());
        reverse_map.insert("edit".to_string(), "search_replace".to_string());
        param_transforms.insert("search_replace".to_string(), Self::edit_transform());

        // todo -> todo_write
        name_map.insert("todo".to_string(), "todo_write".to_string());
        reverse_map.insert("todo_write".to_string(), "todo".to_string());

        // read_file stays same but params change
        param_transforms.insert("read_file".to_string(), Self::read_file_transform());

        // write_file stays same but params change
        param_transforms.insert("write_file".to_string(), Self::write_file_transform());

        Self {
            name_map,
            reverse_map,
            param_transforms,
        }
    }

    /// Create alias config for Qwen-VL models (similar to Coder).
    #[must_use]
    pub fn qwen_vl() -> Self {
        // VL uses same tool names as Coder
        Self::qwen_coder()
    }

    /// Create empty alias config (no transformations).
    #[must_use]
    pub fn none() -> Self {
        Self {
            name_map: HashMap::new(),
            reverse_map: HashMap::new(),
            param_transforms: HashMap::new(),
        }
    }

    /// Get alias config for a model family.
    #[must_use]
    pub fn for_model_family(family: ModelFamily) -> Self {
        match family {
            ModelFamily::QwenCoder => Self::qwen_coder(),
            ModelFamily::QwenVl => Self::qwen_vl(),
            ModelFamily::QwenBase => Self::qwen_coder(), // Base Qwen may also benefit
            _ => Self::none(),
        }
    }

    /// Get alias config for a model name.
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        Self::for_model_family(ModelFamily::from_model_name(model))
    }

    // Parameter transforms for specific tools

    fn bash_transform() -> ParamTransform {
        let mut additional_properties = HashMap::new();

        // Add is_background parameter
        additional_properties.insert(
            "is_background".to_string(),
            json!({
                "type": "boolean",
                "description": "Whether to run the command in background. Default is false. Set to true for long-running processes like development servers, watchers, or daemons.",
                "default": false
            }),
        );

        // Add description parameter
        additional_properties.insert(
            "description".to_string(),
            json!({
                "type": "string",
                "description": "Brief description of what the command does for the user. Be specific and concise."
            }),
        );

        // Add directory parameter
        additional_properties.insert(
            "directory".to_string(),
            json!({
                "type": "string",
                "description": "The absolute path of the directory to run the command in. If not provided, uses project root."
            }),
        );

        ParamTransform {
            renames: HashMap::new(),
            defaults: [("is_background".to_string(), json!(false))]
                .into_iter()
                .collect(),
            additional_required: vec!["is_background".to_string()],
            additional_properties,
        }
    }

    fn read_file_transform() -> ParamTransform {
        let mut renames = HashMap::new();
        // path -> absolute_path
        renames.insert("path".to_string(), "absolute_path".to_string());

        ParamTransform {
            renames,
            ..Default::default()
        }
    }

    fn write_file_transform() -> ParamTransform {
        let mut renames = HashMap::new();
        // path -> file_path
        renames.insert("path".to_string(), "file_path".to_string());

        ParamTransform {
            renames,
            ..Default::default()
        }
    }

    fn edit_transform() -> ParamTransform {
        let mut renames = HashMap::new();
        // file_path -> path (edit uses path)
        renames.insert("file_path".to_string(), "path".to_string());

        // For search_replace -> edit, the content format is different
        // search_replace uses SEARCH/REPLACE blocks in content
        // edit uses old_string/new_string parameters
        // This is a more complex transform handled separately

        let mut additional_properties = HashMap::new();
        additional_properties.insert(
            "old_string".to_string(),
            json!({
                "type": "string",
                "description": "The exact text to find and replace. Must match exactly."
            }),
        );
        additional_properties.insert(
            "new_string".to_string(),
            json!({
                "type": "string",
                "description": "The text to replace old_string with."
            }),
        );

        ParamTransform {
            renames,
            additional_properties,
            additional_required: vec!["old_string".to_string(), "new_string".to_string()],
            ..Default::default()
        }
    }

    /// Get the model-expected name for an internal tool name.
    #[must_use]
    pub fn get_alias<'a>(&'a self, internal_name: &'a str) -> &'a str {
        self.name_map
            .get(internal_name)
            .map(String::as_str)
            .unwrap_or(internal_name)
    }

    /// Get the internal name from a model tool name.
    #[must_use]
    pub fn get_internal_name<'a>(&'a self, model_name: &'a str) -> &'a str {
        self.reverse_map
            .get(model_name)
            .map(String::as_str)
            .unwrap_or(model_name)
    }

    /// Check if a tool name needs aliasing.
    #[must_use]
    pub fn has_alias(&self, internal_name: &str) -> bool {
        self.name_map.contains_key(internal_name)
    }

    /// Transform tool arguments from internal format to model format.
    #[must_use]
    pub fn transform_args(&self, tool_name: &str, args: &Value) -> Value {
        let transform = match self.param_transforms.get(tool_name) {
            Some(t) => t,
            None => return args.clone(),
        };

        let mut result = match args {
            Value::Object(map) => map.clone(),
            _ => return args.clone(),
        };

        // Apply renames
        for (old_name, new_name) in &transform.renames {
            if let Some(value) = result.remove(old_name) {
                result.insert(new_name.clone(), value);
            }
        }

        // Apply defaults for missing values
        for (key, default_value) in &transform.defaults {
            if !result.contains_key(key) {
                result.insert(key.clone(), default_value.clone());
            }
        }

        Value::Object(result)
    }

    /// Transform arguments from model format back to internal format.
    #[must_use]
    pub fn reverse_transform_args(&self, tool_name: &str, args: &Value) -> Value {
        let internal_name = self.get_internal_name(tool_name);
        let transform = match self.param_transforms.get(internal_name) {
            Some(t) => t,
            None => return args.clone(),
        };

        let mut result = match args {
            Value::Object(map) => map.clone(),
            _ => return args.clone(),
        };

        // Reverse renames
        for (internal_name, model_name) in &transform.renames {
            if let Some(value) = result.remove(model_name) {
                result.insert(internal_name.clone(), value);
            }
        }

        Value::Object(result)
    }

    /// Transform a tool's parameter schema for model compatibility.
    #[must_use]
    pub fn transform_schema(&self, tool_name: &str, schema: &Value) -> Value {
        let transform = match self.param_transforms.get(tool_name) {
            Some(t) => t,
            None => return schema.clone(),
        };

        let mut result = schema.clone();

        if let Some(obj) = result.as_object_mut() {
            // Transform properties
            if let Some(Value::Object(props)) = obj.get_mut("properties") {
                // Rename existing properties
                let mut new_props = Map::new();
                for (key, value) in props.iter() {
                    let new_key = transform
                        .renames
                        .get(key)
                        .cloned()
                        .unwrap_or_else(|| key.clone());
                    new_props.insert(new_key, value.clone());
                }

                // Add additional properties
                for (key, value) in &transform.additional_properties {
                    if !new_props.contains_key(key) {
                        new_props.insert(key.clone(), value.clone());
                    }
                }

                *props = new_props;
            }

            // Update required array
            if let Some(Value::Array(required)) = obj.get_mut("required") {
                // Rename required fields
                for item in required.iter_mut() {
                    if let Value::String(s) = item
                        && let Some(new_name) = transform.renames.get(s)
                    {
                        *s = new_name.clone();
                    }
                }

                // Add additional required fields
                for field in &transform.additional_required {
                    if !required.iter().any(|v| v.as_str() == Some(field)) {
                        required.push(Value::String(field.clone()));
                    }
                }
            }
        }

        result
    }
}

/// Tool alias manager that handles bidirectional name mapping.
#[derive(Debug, Clone)]
pub struct ToolAliasManager {
    config: ToolAliasConfig,
    model_family: ModelFamily,
}

impl ToolAliasManager {
    /// Create a new alias manager for a model.
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        let model_family = ModelFamily::from_model_name(model);
        Self {
            config: ToolAliasConfig::for_model_family(model_family),
            model_family,
        }
    }

    /// Create alias manager for a model family.
    #[must_use]
    pub fn for_family(family: ModelFamily) -> Self {
        Self {
            config: ToolAliasConfig::for_model_family(family),
            model_family: family,
        }
    }

    /// Get the model family.
    #[must_use]
    pub fn model_family(&self) -> ModelFamily {
        self.model_family
    }

    /// Check if aliasing is active for this manager.
    #[must_use]
    pub fn is_active(&self) -> bool {
        !self.config.name_map.is_empty()
    }

    /// Convert internal tool name to model-expected name.
    #[must_use]
    pub fn to_model_name<'a>(&'a self, internal_name: &'a str) -> &'a str {
        self.config.get_alias(internal_name)
    }

    /// Convert model tool name to internal name.
    #[must_use]
    pub fn to_internal_name<'a>(&'a self, model_name: &'a str) -> &'a str {
        self.config.get_internal_name(model_name)
    }

    /// Transform arguments for sending to model.
    #[must_use]
    pub fn to_model_args(&self, tool_name: &str, args: &Value) -> Value {
        self.config.transform_args(tool_name, args)
    }

    /// Transform arguments received from model to internal format.
    #[must_use]
    pub fn from_model_args(&self, model_tool_name: &str, args: &Value) -> Value {
        self.config.reverse_transform_args(model_tool_name, args)
    }

    /// Transform schema for model consumption.
    #[must_use]
    pub fn to_model_schema(&self, tool_name: &str, schema: &Value) -> Value {
        self.config.transform_schema(tool_name, schema)
    }

    /// Get all tool name mappings.
    #[must_use]
    pub fn get_all_mappings(&self) -> &HashMap<String, String> {
        &self.config.name_map
    }
}

/// Convenience function to get aliased tool name.
#[must_use]
pub fn get_tool_alias(model: &str, tool_name: &str) -> String {
    let manager = ToolAliasManager::for_model(model);
    manager.to_model_name(tool_name).to_string()
}

/// Convenience function to get internal tool name from model name.
#[must_use]
pub fn get_internal_tool_name(model: &str, model_tool_name: &str) -> String {
    let manager = ToolAliasManager::for_model(model);
    manager.to_internal_name(model_tool_name).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_coder_aliases() {
        let config = ToolAliasConfig::qwen_coder();

        assert_eq!(config.get_alias("bash"), "run_shell_command");
        assert_eq!(config.get_alias("grep"), "grep_search");
        assert_eq!(config.get_alias("search_replace"), "edit");
        assert_eq!(config.get_alias("todo"), "todo_write");
        assert_eq!(config.get_alias("read_file"), "read_file"); // unchanged
    }

    #[test]
    fn test_reverse_aliases() {
        let config = ToolAliasConfig::qwen_coder();

        assert_eq!(config.get_internal_name("run_shell_command"), "bash");
        assert_eq!(config.get_internal_name("grep_search"), "grep");
        assert_eq!(config.get_internal_name("edit"), "search_replace");
        assert_eq!(config.get_internal_name("todo_write"), "todo");
    }

    #[test]
    fn test_no_aliases_for_generic() {
        let config = ToolAliasConfig::for_model_family(ModelFamily::Generic);

        assert_eq!(config.get_alias("bash"), "bash");
        assert_eq!(config.get_alias("grep"), "grep");
    }

    #[test]
    fn test_bash_arg_transform() {
        let config = ToolAliasConfig::qwen_coder();

        let args = json!({"command": "ls -la"});
        let transformed = config.transform_args("bash", &args);

        // Should add is_background default
        assert_eq!(
            transformed.get("command").and_then(|v| v.as_str()),
            Some("ls -la")
        );
        assert_eq!(
            transformed.get("is_background").and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn test_read_file_arg_transform() {
        let config = ToolAliasConfig::qwen_coder();

        let args = json!({"path": "/home/user/file.txt", "limit": 100});
        let transformed = config.transform_args("read_file", &args);

        // path should become absolute_path
        assert_eq!(
            transformed.get("absolute_path").and_then(|v| v.as_str()),
            Some("/home/user/file.txt")
        );
        assert!(transformed.get("path").is_none());
        assert_eq!(transformed.get("limit").and_then(|v| v.as_i64()), Some(100));
    }

    #[test]
    fn test_reverse_arg_transform() {
        let config = ToolAliasConfig::qwen_coder();

        // Simulate receiving args from model in model format
        let model_args = json!({"absolute_path": "/home/user/file.txt", "limit": 50});
        let internal_args = config.reverse_transform_args("read_file", &model_args);

        // absolute_path should become path
        assert_eq!(
            internal_args.get("path").and_then(|v| v.as_str()),
            Some("/home/user/file.txt")
        );
    }

    #[test]
    fn test_schema_transform() {
        let config = ToolAliasConfig::qwen_coder();

        let schema = json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute"
                }
            },
            "required": ["command"]
        });

        let transformed = config.transform_schema("bash", &schema);

        // Should have is_background added
        let props = transformed.get("properties").unwrap();
        assert!(props.get("is_background").is_some());
        assert!(props.get("description").is_some());
        assert!(props.get("directory").is_some());

        // Required should include is_background
        let required = transformed.get("required").unwrap().as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("is_background")));
    }

    #[test]
    fn test_alias_manager() {
        let manager = ToolAliasManager::for_model("qwen3-coder");

        assert!(manager.is_active());
        assert_eq!(manager.model_family(), ModelFamily::QwenCoder);
        assert_eq!(manager.to_model_name("bash"), "run_shell_command");
        assert_eq!(manager.to_internal_name("grep_search"), "grep");
    }

    #[test]
    fn test_convenience_functions() {
        assert_eq!(get_tool_alias("qwen3-next", "bash"), "run_shell_command");
        assert_eq!(
            get_internal_tool_name("qwen3-coder", "edit"),
            "search_replace"
        );

        // Non-Qwen models shouldn't have aliases
        assert_eq!(get_tool_alias("gpt-4", "bash"), "bash");
    }

    #[test]
    fn test_write_file_transform() {
        let config = ToolAliasConfig::qwen_coder();

        let args = json!({"path": "/home/user/file.txt", "content": "hello"});
        let transformed = config.transform_args("write_file", &args);

        assert_eq!(
            transformed.get("file_path").and_then(|v| v.as_str()),
            Some("/home/user/file.txt")
        );
        assert!(transformed.get("path").is_none());
    }
}
