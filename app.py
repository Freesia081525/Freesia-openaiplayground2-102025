# responses_api_runner_streamlit.py
# Streamlit app additions: Responses API Runner + Multi-Agent Orchestrator + Themed UI
# - Features:
#   * Direct Responses API runner (JSON or id+version)
#   * Multi-agent orchestrator: select N agents from agents.yaml, reorder, edit prompts/params/models, run sequentially
#   * Editable handoff: user can modify each agent's output before passing to the next agent
#   * Theme picker with several curated themes for HF Spaces UI
#   * Robust fallbacks and helpful messages for Hugging Face Spaces deployment

import os
import json
import time
from typing import Optional, Any, Dict, List

import streamlit as st
import yaml

# ------------------ Config / Utilities ------------------
APP_TITLE = "Responses API Runner + Multi-Agent Orchestrator"

# Attempt to import new OpenAI client (Responses API)
OpenAIClientClass = None
try:
    mod_openai_new = __import__("openai", fromlist=["OpenAI"])  # may raise
    OpenAIClientClass = getattr(mod_openai_new, "OpenAI", None)
except Exception:
    OpenAIClientClass = None

# Legacy openai fallback
legacy_openai = None
try:
    import openai as legacy_openai
except Exception:
    legacy_openai = None


def init_session_state():
    st.session_state.setdefault("responses_logs", [])
    st.session_state.setdefault("runs_history", [])
    st.session_state.setdefault("agents_yaml_text", None)
    st.session_state.setdefault("agents_obj", None)
    st.session_state.setdefault("theme", "Blue sky")


def log_event(event_type: str, message: str, meta: Optional[Dict[str, Any]] = None) -> None:
    logs = st.session_state.get("responses_logs", [])
    logs.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "event": event_type, "message": message, "meta": meta})
    st.session_state["responses_logs"] = logs


def create_openai_responses_client(api_key: Optional[str]):
    if not OpenAIClientClass:
        return None
    try:
        if api_key:
            try:
                client = OpenAIClientClass(api_key=api_key)
            except TypeError:
                os.environ["OPENAI_API_KEY"] = api_key
                client = OpenAIClientClass()
        else:
            client = OpenAIClientClass()
        return client
    except Exception as e:
        st.warning(f"Failed to instantiate Responses client: {e}")
        return None


def call_openai_responses_api(client, payload: dict) -> dict:
    try:
        resp = None
        try:
            resp = client.responses.create(**payload)
        except TypeError:
            resp = client.responses.create(prompt=payload)
        except Exception:
            try:
                resp = client.responses.create(payload)
            except Exception as e:
                raise e

        if hasattr(resp, "to_dict"):
            return resp.to_dict()
        try:
            return json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            return {"raw": str(resp)}
    except Exception as e:
        return {"error": str(e)}


def call_legacy_openai(api_key: Optional[str], model: str, input_text: str) -> dict:
    if legacy_openai is None:
        return {"error": "legacy openai package not available"}
    if api_key:
        legacy_openai.api_key = api_key
    try:
        if hasattr(legacy_openai, "ChatCompletion"):
            resp = legacy_openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": input_text}])
            text = ""
            try:
                for choice in resp.get("choices", []):
                    text += choice.get("message", {}).get("content", "")
            except Exception:
                text = str(resp)
            return {"text": text, "raw": resp}
        else:
            resp = legacy_openai.Completion.create(model=model, prompt=input_text)
            text = "
".join([c.get("text", "") for c in resp.get("choices", [])])
            return {"text": text, "raw": resp}
    except Exception as e:
        return {"error": str(e)}


def extract_best_text(result: Any) -> Optional[str]:
    try:
        if not isinstance(result, dict):
            return None
        if "output" in result and isinstance(result["output"], list):
            texts = []
            for o in result["output"]:
                if isinstance(o, dict) and "content" in o:
                    if isinstance(o["content"], list):
                        for c in o["content"]:
                            if isinstance(c, dict) and c.get("type") == "output_text":
                                texts.append(c.get("text") or c.get("content") or "")
                            elif isinstance(c, str):
                                texts.append(c)
                    elif isinstance(o["content"], str):
                        texts.append(o["content"])
                elif isinstance(o, str):
                    texts.append(o)
            return "

".join([t for t in texts if t]) or None
        if "text" in result and isinstance(result["text"], str):
            return result["text"]
        if "choices" in result and isinstance(result["choices"], list):
            out = []
            for c in result["choices"]:
                if isinstance(c, dict) and "text" in c:
                    out.append(c.get("text", ""))
                elif isinstance(c, dict) and "message" in c:
                    out.append(c["message"].get("content", ""))
            return "
".join(out) if out else None
    except Exception:
        return None
    return None


def load_agents_yaml_from_file(uploaded_file) -> Optional[Dict]:
    try:
        if uploaded_file is None:
            return None
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        obj = yaml.safe_load(content)
        st.session_state["agents_yaml_text"] = content
        st.session_state["agents_obj"] = obj
        return obj
    except Exception as e:
        st.warning(f"Failed to load agents.yaml: {e}")
        return None


# ------------------ Themes ------------------
THEMES_CSS = {
    "Blue sky": "body { background: linear-gradient(120deg,#c6f0ff,#f0fbff); color:#04293a } .card{background:rgba(255,255,255,0.85); padding:12px; border-radius:12px}",
    "Snow white": "body{background:#ffffff; color:#111827} .card{background:#fbfcfd; padding:12px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.04)}",
    "Alts.Forest": "body{background:linear-gradient(120deg,#e8f5e9,#ecfdf3); color:#14362b} .card{background:rgba(255,255,255,0.9); padding:12px; border-radius:12px}",
    "Flora": "body{background:linear-gradient(120deg,#fff7ed,#fffaf0); color:#3b2f2f} .card{background:rgba(255,255,255,0.92); padding:14px; border-radius:14px}",
    "Sparkling galaxy": "body{background:radial-gradient(circle at 10% 10%, #0f172a,#03040f); color:#e6f7ff} .card{background:rgba(255,255,255,0.04); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.06)}",
    "Fendi Casa": "body{background:linear-gradient(120deg,#f3efe4,#fffaf5); color:#1f1a17} .card{background:#fff; padding:16px; border-radius:8px; box-shadow:0 6px 18px rgba(0,0,0,0.06)}",
    "Ferrari Sports car": "body{background:linear-gradient(120deg,#2b2b2b,#111111); color:#ffebee} .card{background:rgba(0,0,0,0.6); padding:14px; border-radius:12px; border:1px solid rgba(255,0,0,0.08)}",
}


def apply_theme(theme_name: str):
    css = THEMES_CSS.get(theme_name)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ------------------ Multi-Agent Orchestrator ------------------

def render_agent_editor(agent_def: Dict, idx: int):
    st.markdown(f"### Agent {idx+1}: {agent_def.get('name') or agent_def.get('id')}")
    col1, col2 = st.columns([2,1])
    with col1:
        prompt = st.text_area(f"Prompt (agent_{idx})", value=json.dumps(agent_def.get('default_prompt') or agent_def.get('prompt') or "", ensure_ascii=False), height=120, key=f"agent_{idx}_prompt")
    with col2:
        model = st.selectbox(f"Model (agent_{idx})", options=["gpt-5-nano", "gpt-4.1-mini", "gemini-4o-mini"], index=0, key=f"agent_{idx}_model")
        temp = st.number_input(f"Temp (agent_{idx})", min_value=0.0, max_value=2.0, value=0.2, step=0.1, key=f"agent_{idx}_temp")
        max_tokens = st.number_input(f"Max tokens (agent_{idx})", min_value=16, max_value=8192, value=512, step=16, key=f"agent_{idx}_max_tokens")
    return {
        "prompt": prompt,
        "model": model,
        "temp": temp,
        "max_tokens": int(max_tokens),
        "raw_def": agent_def,
    }


def orchestrator_tab():
    st.header("🤖 Multi-Agent Orchestrator")
    st.markdown("Load `agents.yaml`, pick how many agents to run (or select specific agents), reorder them, edit prompts/params, and run sequentially. After each agent runs you can edit the output before sending to the next agent.")

    st.sidebar.markdown("## Orchestrator settings")
    openai_key = st.sidebar.text_input("OpenAI API key (optional)", type="password", key="orchestrator_api_key")
    agents_file = st.sidebar.file_uploader("Upload agents.yaml", type=["yaml","yml"], key="upload_agents_orch")
    if agents_file:
        agents_obj = load_agents_yaml_from_file(agents_file)
        if agents_obj:
            st.success("agents.yaml loaded")
    else:
        agents_obj = st.session_state.get("agents_obj")

    if not agents_obj:
        st.warning("No agents.yaml loaded. Please upload agents.yaml in the sidebar to use orchestrator.")
        return

    # Flatten agents list
    agents_list = agents_obj.get("agents") if isinstance(agents_obj, dict) else agents_obj
    if not agents_list or not isinstance(agents_list, list):
        st.error("agents.yaml missing top-level 'agents' list or has unexpected format.")
        return

    # User picks how many agents to use and which ones
    st.markdown("#### Select agents to run (order matters) — drag to reorder")
    # We'll present checkboxes and allow reordering via number inputs (simple)
    selected = []
    default_indices = []
    for i, a in enumerate(agents_list):
        name = a.get('name') or a.get('id') or f"agent_{i}"
        checked = st.checkbox(f"Use: {name}", value=False, key=f"use_agent_{i}")
        if checked:
            priority = st.number_input(f"Order (smaller runs earlier) for {name}", min_value=1, max_value=100, value=i+1, key=f"order_agent_{i}")
            selected.append((int(priority), i))
    if not selected:
        st.info("Select at least one agent to orchestrate.")
        return
    # sort by order
    selected_sorted = [idx for _, idx in sorted(selected, key=lambda x: x[0])]

    st.markdown("---")
    st.markdown("## Configure selected agents")
    agent_configs = []
    for seq_idx, ai in enumerate(selected_sorted):
        adef = agents_list[ai]
        cfg = render_agent_editor(adef, seq_idx)
        agent_configs.append(cfg)

    # Option: initial input
    st.markdown("---")
    initial_input = st.text_area("Initial input/context for the first agent (you can leave blank)", value="", height=120, key="orch_initial_input")

    # Run orchestration
    run_orch = st.button("Run orchestration — execute agents sequentially")

    client = create_openai_responses_client(openai_key or os.environ.get("OPENAI_API_KEY"))

    if run_orch:
        st.info("Starting orchestration...")
        current_input = initial_input
        run_record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "agents": []}
        for idx, cfg in enumerate(agent_configs):
            st.markdown(f"### Running agent {idx+1} — {cfg['raw_def'].get('name') or cfg['raw_def'].get('id')}")
            # Decide payload for Responses API or fallback
            # We'll construct a simple prompt payload: model+input
            payload = {}
            # Build a 'prompt' object with content if using Responses API
            prompt_obj = {"content": [{"type": "input_text", "text": cfg['prompt'] if not current_input else f"Context:
{current_input}

Prompt:
{cfg['prompt']}"}]} 
            payload['prompt'] = prompt_obj
            payload['model'] = cfg['model']
            # include temperature/max tokens where supported
            payload.setdefault('temperature', cfg['temp'])
            payload.setdefault('max_output_tokens', cfg['max_tokens'])

            # Try Responses client first
            result = None
            if client:
                with st.spinner(f"Calling Responses API for agent {idx+1}..."):
                    result = call_openai_responses_api(client, payload)
                    st.json(result)
                    text = extract_best_text(result)
            else:
                # fallback: legacy API using textual prompt
                with st.spinner("Calling legacy OpenAI API fallback..."):
                    textual = (f"Context:
{current_input}

Prompt:
{cfg['prompt']}") if current_input else cfg['prompt']
                    legacy = call_legacy_openai(openai_key or os.environ.get("OPENAI_API_KEY"), cfg['model'], textual)
                    if 'error' in legacy:
                        st.error(f"Legacy call failed: {legacy['error']}")
                        text = None
                        result = legacy
                    else:
                        st.json(legacy.get('raw'))
                        text = legacy.get('text')
                        result = legacy.get('raw')

            # Show extracted text and allow user to edit before passing to next
            extracted = text if (text is not None) else json.dumps(result)[:2000]
            st.markdown("**Extracted output (editable) — edit before passing to next agent:**")
            edited = st.text_area(f"agent_{idx}_edited_output", value=extracted or "", height=160, key=f"agent_{idx}_edited")

            # Save run detail
            run_record['agents'].append({
                'agent_def': cfg['raw_def'],
                'payload': payload,
                'raw_result_preview': str(result)[:1000],
                'extracted': extracted,
                'edited': edited,
            })

            # Set current_input for next agent
            current_input = edited
            log_event("orchestrator_agent_run", f"Ran agent {idx+1}", {'agent': cfg['raw_def'].get('id') or cfg['raw_def'].get('name'), 'preview': str(extracted)[:400]})

        # Save run
        st.success("Orchestration complete — results saved to session history")
        st.session_state['runs_history'].append(run_record)
        st.download_button("Download run JSON", data=json.dumps(run_record, ensure_ascii=False, indent=2), file_name=f"orchestrator_run_{int(time.time())}.json")


# ------------------ Responses API Runner Tab (improved) ------------------

def responses_api_runner_tab():
    st.header("📡 Direct Responses API Runner")
    st.markdown("Paste a `prompt` payload JSON or provide a prompt id+version. Optionally save runs and feed into Orchestrator.")

    st.sidebar.markdown("## Responses Runner settings")
    openai_key = st.sidebar.text_input("OpenAI API key (optional)", type="password", key="responses_api_key")

    run_mode = st.radio("Input mode", ["JSON payload", "Prompt id + version"], index=0)
    client = create_openai_responses_client(openai_key or os.environ.get("OPENAI_API_KEY"))

    prompt_payload = None
    if run_mode == "JSON payload":
        prompt_json_text = st.text_area("Prompt JSON", height=160, value='{}')
        try:
            prompt_payload = json.loads(prompt_json_text) if prompt_json_text.strip() else None
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            prompt_payload = None
    else:
        prompt_id = st.text_input("Prompt id", value="")
        prompt_version = st.text_input("Prompt version", value="1")
        if prompt_id.strip():
            prompt_payload = {"id": prompt_id.strip(), "version": prompt_version.strip()}

    save_run = st.checkbox("Save run to history", value=True)
    auto_feed_to_orch = st.checkbox("After run, offer to open Orchestrator with this output", value=True)

    if st.button("Run Responses API"):
        if not prompt_payload:
            st.error("No prompt payload provided.")
        else:
            if client:
                st.info("Calling Responses API...")
                payload = dict(prompt_payload) if isinstance(prompt_payload, dict) else {"prompt": prompt_payload}
                # if no model in payload, attach a default
                payload.setdefault('model', 'gpt-5-nano')
                result = call_openai_responses_api(client, payload)
                st.markdown("#### Raw response")
                st.json(result)
                text = extract_best_text(result)
                if text:
                    st.markdown("#### Extracted text")
                    st.code(text)
                if save_run:
                    st.session_state['runs_history'].append({'ts': time.strftime("%Y-%m-%d %H:%M:%S"), 'payload': payload, 'result': result})
                if auto_feed_to_orch and text:
                    if st.button("Open orchestrator with this output"):
                        st.session_state['orch_initial_text'] = text
                        st.experimental_rerun()
            else:
                st.warning("Responses client not available; trying legacy fallback if installed.")
                textual = json.dumps(prompt_payload) if isinstance(prompt_payload, dict) else str(prompt_payload)
                fallback = call_legacy_openai(openai_key or os.environ.get("OPENAI_API_KEY"), 'gpt-4.1-mini', textual)
                if 'error' in fallback:
                    st.error(f"Legacy call failed: {fallback['error']}")
                else:
                    st.code(fallback.get('text'))
                    st.session_state['runs_history'].append({'ts': time.strftime("%Y-%m-%d %H:%M:%S"), 'payload': prompt_payload, 'result': fallback})


# ------------------ Main App ------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()

    st.title(APP_TITLE)

    # Theme selector
    with st.sidebar.expander("Theme & UI"):
        theme = st.selectbox("Theme", options=list(THEMES_CSS.keys()), index=list(THEMES_CSS.keys()).index(st.session_state.get('theme', 'Blue sky')))
        st.session_state['theme'] = theme
        apply_theme(theme)
        st.markdown("---")
        st.markdown("**UI tips:** Use the Orchestrator tab to chain agents. Save runs for auditability.")

    # Tabs
    tabs = st.tabs(["Responses Runner", "Orchestrator (multi-agent)", "Activity & History"])
    with tabs[0]:
        responses_api_runner_tab()
    with tabs[1]:
        orchestrator_tab()
    with tabs[2]:
        st.header("Activity & Run History")
        st.markdown("Recent events and saved runs from this session.")
        logs = st.session_state.get('responses_logs', [])
        st.subheader("Events log")
        for e in logs[::-1][:50]:
            st.write(e)
        st.markdown("---")
        st.subheader("Runs history")
        runs = st.session_state.get('runs_history', [])
        st.write(f"{len(runs)} runs in session")
        for i, r in enumerate(runs[::-1]):
            with st.expander(f"Run #{len(runs)-i} — {r.get('ts')}"):
                st.json(r)
                st.download_button("Download run", data=json.dumps(r, ensure_ascii=False, indent=2), file_name=f"run_{i}.json")


if __name__ == '__main__':
    main()
