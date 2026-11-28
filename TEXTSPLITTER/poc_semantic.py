from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    HuggingFaceEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

text="""
After installing the Windsurf plugin in VS Code, the main feature you use is Windsurf Chat, which allows you to converse with your entire codebase directly within the editor.

Location and Access: By default, it's located in the left sidebar of VS Code. You can quickly open the chat panel using the shortcut Ctrl+Shift+A (Windows/Linux) or Cmd+Shift+A (Mac).

Core Capability: It's powered by Windsurf's contextual awareness engine, enabling it to understand your code and provide answers. You can reference specific files, functions, or classes by typing @ in the chat, or use the /explain command to have it explain code snippets.

Context Management: You can "pin" files or code snippets in the chat panel to ensure the AI always considers this context in subsequent conversations, or set persistent personalized instructions.

📦 Plugin Background and Current Status
To better understand it, you need to know its origins and current situation:

Historical Origin: The Windsurf plugin was formerly known as Codeium, a free AI coding assistant. After the company launched the standalone Windsurf editor, it renamed the Codeium extension to the Windsurf plugin.

Current Status: According to information from late 2025, the Windsurf plugin for VS Code is currently in "maintenance mode". This means it may not receive major feature updates anymore. The Windsurf team's focus has shifted to their standalone AI IDE.

Relationship with the Standalone Editor: Many of the powerful features people associate with "Windsurf" (such as the Cascade agent that can autonomously edit multiple files and run terminal commands) are core features of the standalone Windsurf editor, not functionality provided by the VS Code plugin.

💡 Simple Summary
If you just want to chat with AI in VS Code to ask questions about your code or get explanations, the Windsurf plugin is a lightweight choice. But if you're interested in Windsurf's advertised autonomous multi-file editing, running commands, and other agent-level capabilities, you need to download and use the standalone Windsurf editor (which is itself built on top of VS Code).
"""

result = text_splitter.create_documents([text])

for chunks in result:
    print(chunks, "\n")
