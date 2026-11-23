package gateway

import (
	"testing"
)

// TestResponsesToChat_ModelSwitchDeveloperContext 模拟 Codex 切换模型时注入 developer 消息的真实会话场景：
// instructions (system) -> user 1 -> assistant 1 -> developer (时间/配置更新) -> user 2。
// 转换后的 Chat 请求中必须只有首部 1 条 system 消息，中间的 developer 消息必须折叠到 user 2，
// 从而满足通义千问等模型「仅允许首部一条 system 消息且不得在对话中途插入 system」的严格要求。
func TestResponsesToChat_ModelSwitchDeveloperContext(t *testing.T) {
	in := map[string]any{
		"model":        "qwenwork/qwen4-max",
		"instructions": "You are Codex, a coding agent.",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Turn 1 request"},
				},
			},
			map[string]any{
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "output_text", "text": "Turn 1 response"},
				},
			},
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<codex_apps_client_time_context>2026-11-23</codex_apps_client_time_context>"},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Turn 2 after switch model"},
				},
			},
		},
	}

	req, err := responsesToChat(in, "qwenwork/qwen4-max", 0)
	if err != nil {
		t.Fatalf("responsesToChat failed: %v", err)
	}

	msgs, ok := req["messages"].([]any)
	if !ok {
		t.Fatalf("req[messages] missing or not slice: %#v", req)
	}

	if len(msgs) != 4 {
		t.Fatalf("expected 4 messages, got %d: %#v", len(msgs), msgs)
	}

	// 1. 验证角色序列：严格为 system -> user -> assistant -> user
	expectedRoles := []string{"system", "user", "assistant", "user"}
	for i, wantRole := range expectedRoles {
		m, ok := msgs[i].(map[string]any)
		if !ok || m["role"] != wantRole {
			t.Fatalf("msgs[%d] role = %v, want %s", i, m["role"], wantRole)
		}
	}

	// 2. 验证首部 system 提示保留了 instructions
	m0, _ := msgs[0].(map[string]any)
	if m0["content"] != "You are Codex, a coding agent." {
		t.Fatalf("m0 content = %q", m0["content"])
	}

	// 3. 验证 user 2 中包含了中途插入的 developer 时间上下文
	m3, _ := msgs[3].(map[string]any)
	contentStr, _ := m3["content"].(string)
	if contentStr != "<codex_apps_client_time_context>2026-11-23</codex_apps_client_time_context>\n\nTurn 2 after switch model" {
		t.Fatalf("user 2 content = %q", contentStr)
	}
}

// TestResponsesToChat_EmptyToolsDropsToolChoice 验证：当 tools 列表为空或只包含非 function 类型被过滤后，
// tool_choice 必须被剔除，避免上游因 tools 为空却传 tool_choice 判 400 参数非法。
func TestResponsesToChat_EmptyToolsDropsToolChoice(t *testing.T) {
	in := map[string]any{
		"model":       "qwenwork/qwen4-max",
		"tool_choice": "auto",
		"tools": []any{
			map[string]any{"type": "web_search", "name": "search"}, // 非 function 工具会被跳过
		},
		"input": "hello",
	}

	req, err := responsesToChat(in, "qwenwork/qwen4-max", 0)
	if err != nil {
		t.Fatalf("responsesToChat failed: %v", err)
	}

	if _, hasTools := req["tools"]; hasTools {
		t.Fatalf("tools should not be present when only non-function tools were passed")
	}
	if tc, hasTC := req["tool_choice"]; hasTC {
		t.Fatalf("tool_choice should be dropped when tools is empty, got %v", tc)
	}
}

// TestResponsesToChat_OrphanFunctionCallCleaned 验证：客户端之前一轮异常中断导致只有 function_call
// 而没有对应的 function_call_output 时，孤儿工具调用能自愈并剔除空 assistant 消息。
func TestResponsesToChat_OrphanFunctionCallCleaned(t *testing.T) {
	in := map[string]any{
		"model": "qwenwork/qwen4-max",
		"input": []any{
			map[string]any{
				"type": "message", "role": "user", "content": "turn 1",
			},
			map[string]any{
				"type": "function_call", "call_id": "call_orphan", "name": "do_something", "arguments": "{}",
			},
			map[string]any{
				"type": "message", "role": "user", "content": "turn 2",
			},
		},
	}

	req, err := responsesToChat(in, "qwenwork/qwen4-max", 0)
	if err != nil {
		t.Fatalf("responsesToChat failed: %v", err)
	}

	msgs, _ := req["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (empty assistant pruned), got %d: %#v", len(msgs), msgs)
	}
	m0, _ := msgs[0].(map[string]any)
	m1, _ := msgs[1].(map[string]any)
	if m0["role"] != "user" || m1["role"] != "user" {
		t.Fatalf("expected two user messages, got %#v, %#v", m0, m1)
	}
}
