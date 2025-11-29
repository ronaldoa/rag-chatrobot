import gradio as gr
import time
from datetime import datetime

# 模拟聊天历史存储
chat_sessions = {}
current_session_id = None


def create_new_chat():
    """创建新的聊天会话"""
    global current_session_id
    session_id = f"chat_{int(time.time())}"
    chat_sessions[session_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    current_session_id = session_id
    return [], get_chat_list(), session_id


def get_chat_list():
    """获取聊天列表用于侧边栏显示"""
    chat_list = []
    for session_id, data in sorted(
        chat_sessions.items(), key=lambda x: x[1]["created_at"], reverse=True
    ):
        title = data["title"][:30] + "..." if len(data["title"]) > 30 else data["title"]
        chat_list.append(f"{title}")
    return chat_list if chat_list else ["No chats."]


def chatbot_response(message, history):
    """生成机器人回复"""
    # 简单的回复逻辑，你可以替换为 API 调用
    if "你好" in message or "hello" in message.lower():
        response = "你好！很高兴见到你。我是 GPT-4 Turbo，有什么可以帮助你的吗？"
    elif "再见" in message or "bye" in message.lower():
        response = "再见！祝你有美好的一天！"
    elif "帮助" in message or "help" in message.lower():
        response = "我可以帮助你回答问题、写代码、创意写作等。请随意提问！"
    elif "什么" in message or "?" in message or "？" in message:
        response = f"关于 '{message}' 这个问题，让我来为你解答...\n\n这是一个示例回复。在实际应用中，你可以在这里接入真实的 AI 模型 API，比如 OpenAI、Claude 或本地模型。"
    else:
        response = f"收到你的消息：「{message}」\n\n我理解你想要讨论这个话题。在实际应用中，这里会调用真实的 AI 模型来生成更智能的回复。"

    # 保存到当前会话
    if current_session_id and current_session_id in chat_sessions:
        if len(chat_sessions[current_session_id]["messages"]) == 0:
            # 用第一条消息作为会话标题
            chat_sessions[current_session_id]["title"] = message[:40]

    return response


def load_chat_session(chat_title):
    """加载选中的聊天会话"""
    global current_session_id
    if not chat_sessions or chat_title == "No chats.":
        return []

    # 根据标题找到对应的会话
    for session_id, data in chat_sessions.items():
        display_title = (
            data["title"][:30] + "..." if len(data["title"]) > 30 else data["title"]
        )
        if display_title == chat_title:
            current_session_id = session_id
            return data["messages"]

    return []


# 创建 Gradio 界面
with gr.Blocks() as demo:
    # 添加自定义 CSS（使用 HTML 方式）
    gr.HTML("""
    <style>
        .gradio-container {
            max-width: 100% !important;
        }
        #chatbot {
            height: 600px;
        }
    </style>
    """)

    gr.Markdown("# 🤖 GPT-4 Turbo Chatbot")

    with gr.Row():
        # 左侧边栏
        with gr.Column(scale=1):
            gr.Markdown("### 💬 Workspace")
            workspace_dropdown = gr.Dropdown(
                choices=["Default Workspace", "Work", "Personal"],
                value="Default Workspace",
                label="Select workspace",
            )

            new_chat_btn = gr.Button("➕ New Chat")

            search_box = gr.Textbox(
                placeholder="Search chats...", label="Search", show_label=False
            )

            gr.Markdown("### 📝 Chat History")
            chat_list = gr.Radio(choices=get_chat_list(), label="", show_label=False)

        # 右侧聊天区域
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=600, elem_id="chatbot")

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Send a message...",
                    show_label=False,
                    container=False,
                    scale=9,
                )
                send_btn = gr.Button("✈️", scale=1)

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat")

            gr.Markdown("---")
            gr.Markdown("**Model:** GPT-4 Turbo | **Quick Settings** ⚙️")

    # 会话 ID 状态
    session_state = gr.State(None)

    # 事件处理函数
    def user_message(user_msg, history):
        return "", history + [[user_msg, None]]

    def bot_message(history):
        if history and history[-1][1] is None:
            user_msg = history[-1][0]
            bot_msg = chatbot_response(user_msg, history)
            history[-1][1] = bot_msg

            # 保存到当前会话
            if current_session_id and current_session_id in chat_sessions:
                chat_sessions[current_session_id]["messages"] = history

        return history, get_chat_list()

    # 绑定事件
    msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
        bot_message, chatbot, [chatbot, chat_list]
    )

    send_btn.click(user_message, [msg, chatbot], [msg, chatbot]).then(
        bot_message, chatbot, [chatbot, chat_list]
    )

    new_chat_btn.click(create_new_chat, None, [chatbot, chat_list, session_state])

    chat_list.change(load_chat_session, chat_list, chatbot)

    clear_btn.click(lambda: [], None, chatbot)

# 启动应用
if __name__ == "__main__":
    # 创建初始会话
    create_new_chat()
    demo.launch()
