from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text = """
After pressing the shortcut, the microphone icon will appear and you can start speaking directly. VS Code will convert what you say into text in real-time and insert it at the cursor position. Stop dictation: Press the Esc key to stop, or press the same shortcut again. Walkie-talkie mode: If you press and hold the shortcut without releasing, VS Code will only listen while you're holding the key, and stop automatically when you release it, which works like a walkie-talkie. First-time use note: When you use it for the first time, VS Code may automatically download a small speech recognition model (only a few hundred MB), after which recognition can work offline locally, and your voice data won't be uploaded to the cloud. If you can't start it by pressing the shortcut, you can open the command palette (Ctrl+Shift+P), search for Voice: Start Dictation in Editor, and execute it to confirm whether the feature is enabled.
"""

splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separator=" "
)

result = splitter.split_text(text)

for ele in result:
    print(ele, "\n")