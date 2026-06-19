from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

full_text = ""
collected_chunks = []

# Streaming: tokens arrive one chunk at a time
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {   
            "role": "user", 
            "content": "List three benefits of streaming APIs"
        }
    ],
    stream=True # <-- this enables streaming
)

# Iterate over each chunk as it arrives
for chunk in stream:
    delta_content = chunk.choices[0].delta.content

    if delta_content:
        full_text = full_text + delta_content
        collected_chunks.append(delta_content)
        print(delta_content, end="", flush=True)

print() # newline after streaming finishes
print(f"\nTotal characters received:{len(full_text)}")
print(f"\nTotal chunks received:{len(collected_chunks)}")
print(collected_chunks)

# Output:
# Total characters received:1280

# Total chunks received:220
# ['Streaming', ' APIs', ' offer', ' several', ' advantages', ',', 
# ' particularly', ' for', ' applications', ' that', ' require', 
# ' real', '-time', ' data', ' processing', ' or', ' updates', '.', 
# ' Here', ' are', ' three', ' key', ' benefits', ':\n\n', '1', '.', 
# ' **', 'Real', '-time', ' Data', ' Access', '**', ':', ' Streaming', 
# ' APIs', ' provide', ' immediate', ' access', ' to', ' data', ' as', 
# ' it', ' is', ' generated', ',', ' allowing', ' applications', ' to', 
# ' respond', ' to', ' events', ' in', ' real', '-time', '.', ' This', 
# ' is', ' crucial', ' for', ' use', ' cases', ' such', ' as', ' social', 
# ' media', ' feeds', ',', ' financial', ' market', ' data', ',', ' or',
#  ' any', ' application', ' that', ' relies', ' on', ' timely', 
# ' information', '.\n\n', '2', '.', ' **', 'Reduced', ' Lat', 'ency', 
# ' and', ' Improved', ' Performance', '**', ':', ' By', ' maintaining', 
# ' an', ' open', ' connection', ' for', ' continuous', ' data', 
# ' transfer', ',', ' streaming', ' APIs', ' minimize', ' the', 
# ' latency', ' that', ' typically', ' comes', ' with', ' traditional', 
# ' request', '-response', ' models', '.', ' This', ' leads', ' to', 
# ' faster', ' data', ' retrieval', ' and', ' enhances', ' overall', 
# ' application', ' performance', ',', ' especially', ' in', ' scenarios', 
# ' where', ' frequent', ' updates', ' are', ' needed', '.\n\n', '3', 
# '.', ' **', 'Eff', 'icient', ' Resource', ' Util', 'ization', '**', 
# ':', ' Streaming', ' APIs', ' can', ' be', ' more', ' efficient', 
# ' than', ' traditional', ' APIs', ',', ' as', ' they', ' reduce', 
# ' the', ' overhead', ' of', ' repeated', ' connections', ' and', 
# ' requests', '.', ' Instead', ' of', ' constantly', ' polling', 
# ' for', ' updates', ',', ' clients', ' receive', ' data', 
# ' directly', ' as', ' it', ' becomes', ' available', ',', 
# ' which', ' can', ' lead', ' to', ' lower', ' bandwidth', 
# ' usage', ' and', ' decreased', ' server', ' load', '.\n\n', 
# 'These', ' benefits', ' make', ' streaming', ' APIs', 
# ' particularly', ' suitable', ' for', ' applications', 
# ' that', ' need', ' to', ' handle', ' large', ' volumes', ' of', 
# ' data', ' with', ' a', ' high', ' frequency', ' and', ' low', ' delay', 
# '.']