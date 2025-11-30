#"""Gradio UI components."""
#
#import gradio as gr
#from src.qa_service import qa_service
#from src.config import Settings
#
#
#def create_gradio_interface():
#    """Create the Gradio chat interface."""
#
#    compact_css = """
#    :root {
#        --sidebar-width: 320px;
#    }
#    #header-title {
#        font-weight: 700;
#        letter-spacing: -0.02em;
#        margin: 6px 0 2px 6px;
#    }
#    #layout-row {
#        gap: 12px;
#    }
#    #sidebar {
#        min-width: var(--sidebar-width);
#        max-width: var(--sidebar-width);
#        border: 1px solid #e5e7eb;
#        background: linear-gradient(145deg, #f8fafc 0%, #eef2ff 100%);
#        padding: 12px;
#        border-radius: 12px;
#        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
#    }
#    #sidebar .compact-controls .gradio-slider {
#        margin-top: 2px;
#        margin-bottom: 6px;
#    }
#    #sidebar .compact-controls .wrap {
#        gap: 8px !important;
#    }
#    #sidebar .compact-controls .gr-box {
#        padding: 8px 10px;
#    }
#    #sidebar-toggle {
#        width: 52px;
#    }
#    #llm-shortcut {
#        width: 100%;
#    }
#    #main-panel {
#        border-radius: 12px;
#        background: #fff;
#        border: 1px solid #e5e7eb;
#        padding: 8px 12px 16px;
#        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
#    }
#    """
#
#    def chat_function(message, history):
#        """Handle chat messages."""
#        history = history or []
#        history.append({"role": "user", "content": message})
#        try:
#            result = qa_service.ask(message)
#            answer = result["answer"]
#
#            # Append source information
#            if result["sources"]:
#                answer += "\n\n---\n📚 **References:**\n"
#                for i, source in enumerate(result["sources"], 1):
#                    answer += f"{i}. `{source['source']}`"
#                    if source.get("page"):
#                        answer += f" (page {source['page']})"
#                    answer += "\n"
#
#            history.append({"role": "assistant", "content": answer})
#        except Exception as e:
#            history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
#
#        return history, ""
#
#    def apply_llm_params(n_ctx, n_threads, n_gpu_layers, n_batch, temperature, top_p, repeat_penalty, max_tokens):
#        """Apply LLM parameter changes and rebuild chain."""
#        try:
#            qa_service.update_llm_params(
#                {
#                    "n_ctx": n_ctx,
#                    "n_threads": n_threads,
#                    "n_gpu_layers": n_gpu_layers,
#                    "n_batch": n_batch,
#                    "temperature": temperature,
#                    "top_p": top_p,
#                    "repeat_penalty": repeat_penalty,
#                    "max_tokens": max_tokens,
#                }
#            )
#            return "✅ LLM parameters updated."
#        except Exception as e:
#            return f"❌ Failed to update: {e}"
#
#    def system_info():
#        """Return merged system + LLM info."""
#        info = {
#            "config": Settings.to_dict(),
#            "llm": qa_service.current_llm_params(),
#        }
#        return info
#
#    with gr.Blocks(title="🦙 Llama 3 Local Knowledge QA") as demo:
#        gr.HTML(f"<style>{compact_css}</style>")
#        sidebar_state = gr.State(True)
#        with gr.Row():
#            toggle_btn = gr.Button("✕", variant="secondary", elem_id="sidebar-toggle")
#            gr.Markdown("## 🦙 Llama 3 Local Knowledge QA", elem_id="header-title")
#        gr.Markdown("**Stack:** LlamaCpp + GGUF + FAISS + Reranker")
#
#        def _dict_to_table(data: dict):
#            return [[k, str(v)] for k, v in data.items()]
#
#        def toggle_sidebar(is_open: bool):
#            new_state = not is_open
#            label = "✕" if new_state else "☰"
#            return new_state, gr.update(visible=new_state), gr.Button.update(value=label)
#
#        with gr.Row(elem_id="layout-row"):
#            # Left control rail
#            with gr.Column(scale=1, elem_id="sidebar", visible=True) as sidebar:
#                gr.Markdown("#### 快捷控制")
#
#                llm_shortcut = gr.Button("LLM Settings", variant="primary", elem_id="llm-shortcut")
#                sys_info_box = gr.JSON(value=system_info(), label="System info")
#
#                with gr.Accordion("LLM Settings", open=True, elem_classes="compact-controls"):
#                    with gr.Row():
#                        n_ctx = gr.Slider(1024, 8192, value=4096, step=512, label="Context window")
#                        max_tokens = gr.Slider(64, 2048, value=512, step=32, label="Max tokens")
#                        n_batch = gr.Slider(32, 1024, value=512, step=32, label="Batch")
#                    with gr.Row():
#                        temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
#                        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
#                        repeat_penalty = gr.Slider(0.8, 1.5, value=1.15, step=0.01, label="Repeat penalty")
#                    with gr.Row():
#                        n_threads = gr.Slider(1, 32, value=8, step=1, label="CPU threads")
#                        n_gpu_layers = gr.Slider(0, 64, value=0, step=1, label="GPU layers")
#
#                    def refresh_llm():
#                        params = qa_service.current_llm_params()
#                        return (
#                            params["n_ctx"],
#                            params["max_tokens"],
#                            params["n_batch"],
#                            params["temperature"],
#                            params["top_p"],
#                            params["repeat_penalty"],
#                            params["n_threads"],
#                            params["n_gpu_layers"],
#                            system_info(),
#                        )
#
#                    load_btn = gr.Button("Load current LLM params", variant="secondary")
#                    load_btn.click(
#                        refresh_llm,
#                        None,
#                        [n_ctx, max_tokens, n_batch, temperature, top_p, repeat_penalty, n_threads, n_gpu_layers, sys_info_box],
#                    )
#
#                    status = gr.Markdown("")
#                    apply_btn = gr.Button("Apply LLM parameters", variant="primary")
#                    apply_btn.click(
#                        apply_llm_params,
#                        inputs=[n_ctx, n_threads, n_gpu_layers, n_batch, temperature, top_p, repeat_penalty, max_tokens],
#                        outputs=status,
#                    )
#
#                sys_btn = gr.Button("Refresh system info", variant="secondary")
#                sys_btn.click(system_info, None, sys_info_box)
#
#                gr.Markdown("---")
#                gr.Markdown("Status")
#
#            # Main content
#            with gr.Column(scale=3, elem_id="main-panel"):
#                tab_state = gr.State("chat")
#
#                def switch_tab(target: str):
#                    return target, gr.update(visible=target == "chat"), gr.update(visible=target == "config")
#
#                with gr.Row():
#                    chat_tab_btn = gr.Button("Chat", variant="secondary")
#                    config_tab_btn = gr.Button("Config", variant="secondary")
#
#                with gr.Column(visible=True) as chat_panel:
#                    chatbot = gr.Chatbot(label="Chat")
#                    msg = gr.Textbox(label="Your question")
#                    clear = gr.Button("Clear")
#                    msg.submit(chat_function, [msg, chatbot], [chatbot, msg])
#                    clear.click(lambda: [], None, chatbot)
#
#                with gr.Column(visible=False) as config_panel:
#                    with gr.Row():
#                        config_box = gr.Dataframe(
#                            headers=["Key", "Value"],
#                            datatype=["str", "str"],
#                            value=_dict_to_table(Settings.to_dict()),
#                            label="Config",
#                            interactive=False,
#                        )
#                        llm_box = gr.Dataframe(
#                            headers=["Key", "Value"],
#                            datatype=["str", "str"],
#                            value=_dict_to_table(qa_service.current_llm_params()),
#                            label="Current LLM params",
#                            interactive=False,
#                        )
#
#                    def refresh_config():
#                        return _dict_to_table(Settings.to_dict())
#
#                    refresh_btn = gr.Button("Refresh config")
#                    refresh_btn.click(refresh_config, None, config_box)
#
#                    def refresh_llm_table():
#                        return _dict_to_table(qa_service.current_llm_params())
#
#                    refresh_llm_btn = gr.Button("Refresh LLM table")
#                    refresh_llm_btn.click(refresh_llm_table, None, llm_box)
#
#                chat_tab_btn.click(lambda: switch_tab("chat"), None, [tab_state, chat_panel, config_panel])
#                config_tab_btn.click(lambda: switch_tab("config"), None, [tab_state, chat_panel, config_panel])
#
#        toggle_btn.click(toggle_sidebar, sidebar_state, [sidebar_state, sidebar, toggle_btn])
#        llm_shortcut.click(lambda: switch_tab("config"), None, [tab_state, chat_panel, config_panel])
#
#    return demo
#