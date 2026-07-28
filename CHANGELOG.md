# Changelog

## 4.0.0

### Breaking

- Require `nimble_python>=1.0.0,<2.0.0` (incompatible with trees pinned to 0.x)
- URL extract paths call `client.extract.run(...)` (SDK 1.x resource API)
- Deprecated `NimbleAgentListTool` no longer accepts `search` / `managed_by` / `privacy` (Extract Templates list has no equivalent filters)
- Deprecated `NimbleAgentListTool` / `NimbleAgentGetTool` return Extract Templates shapes (no legacy `managed_by` / `description` / `is_public` / `input_properties`)
- Toolkit flag `include_web_search_agents` replaces never-released `include_agents`
- Agent API V2 tool names use the `nimble_web_search_agent_*` prefix (see Added)

### Added

- Extract Templates tools: `NimbleExtractTemplateListTool`, `NimbleExtractTemplateGetTool`, `NimbleExtractTemplateRunTool` (`nimble_extract_template_*`)
- Agent API V2 tools (resumable start/status/result):
  - `nimble_web_search_agents_list`
  - `nimble_web_search_agent_templates_list`
  - `nimble_web_search_agent_create`
  - `nimble_web_search_agent_run_start` / `run_status` / `run_result`
- Toolkit flags: `include_extract_templates`, `include_web_search_agents`
- Attribution via SDK `client_source="langchain-nimble"`

### Deprecated

- `NimbleAgentListTool` / `NimbleAgentGetTool` / `NimbleAgentRunTool` wrap Extract Templates and emit `DeprecationWarning`; prefer `nimble_extract_template_*`
- `NimbleToolkit(include_agent=True)` — prefer `include_extract_templates` (or `include_web_search_agents` for research)

### Notes

- Legacy `nimble_agent_*` names are **not** repointed to Agent API V2 research agents
- Agent API V2 tools are resumable; they do not poll inside one call
- Failed Agent API V2 results raise `ToolException` (aligned with Extract Template non-success handling)
