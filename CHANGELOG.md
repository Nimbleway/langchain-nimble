# Changelog

## 3.1.0

### Added

- Extract Templates tools: `NimbleExtractTemplateListTool`, `NimbleExtractTemplateGetTool`, `NimbleExtractTemplateRunTool` (`nimble_extract_template_*`)
- Agent API V2 tools: `NimbleAgentsListTool`, `NimbleAgentTemplatesListTool`, `NimbleAgentCreateTool`, `NimbleAgentRunStartTool`, `NimbleAgentRunStatusTool`, `NimbleAgentRunResultTool`
- Toolkit flags: `include_extract_templates`, `include_agents`
- Attribution via SDK `client_source="langchain-nimble"`

### Changed

- Require `nimble_python>=1.0.0,<2.0.0`
- URL extract paths call `client.extract.run(...)` (SDK 1.x resource API)
- Package version bumped to 3.1.0

### Deprecated

- `NimbleAgentListTool` / `NimbleAgentGetTool` / `NimbleAgentRunTool` now wrap Extract Templates and emit `DeprecationWarning`; prefer `nimble_extract_template_*`
- `NimbleToolkit(include_agent=True)` — prefer `include_extract_templates` (or `include_agents` for research)

### Notes

- Legacy agent tool names are **not** repointed to Agent API V2 research agents
- Agent API V2 tools are resumable (start/status/result); they do not poll inside one call
