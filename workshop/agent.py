import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import MessageRole, FilePurpose, FunctionTool, FileSearchTool, ToolSet
from dotenv import load_dotenv
from tools import calculate_pizza_for_people
from azure.ai.agents.models import MessageRole, FilePurpose, FunctionTool, FileSearchTool, ToolSet, McpTool
import time 

load_dotenv(override=True)

project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_CONNECTION_STRING"],
    credential=DefaultAzureCredential()
)

functions = FunctionTool(functions={calculate_pizza_for_people})

# Upload all files in the documents directory
print(f"Uploading files from ./documents ...")
file_ids = [
    project_client.agents.files.upload_and_poll(file_path=os.path.join("./documents", f), purpose=FilePurpose.AGENTS).id
    for f in os.listdir("./documents")
    if os.path.isfile(os.path.join("./documents", f))
]
print(f"Uploaded {len(file_ids)} files.")

# Create a vector store with the uploaded files
vector_store = project_client.agents.vector_stores.create_and_poll(
    data_sources=[],
    name="contoso-pizza-store-information"
)
print(f"Created vector store, vector store ID: {vector_store.id}")

# Create a vector store file batch to process the uploaded files
batch = project_client.agents.vector_store_file_batches.create_and_poll(
    vector_store_id=vector_store.id,
    file_ids=file_ids
)

file_search = FileSearchTool(vector_store_ids=[vector_store.id])

mcp_tool = McpTool(
    server_label="contoso_pizza",
    server_url="https://ca-pizza-mcp-sc6u2typoxngc.graypond-9d6dd29c.eastus2.azurecontainerapps.io/sse",
    allowed_tools=[
        "get_pizzas",
        "get_pizza_by_id",
        "get_toppings",
        "get_topping_by_id",
        "get_topping_categories",
        "get_orders",
        "get_order_by_id",
        "place_order",
        "delete_order_by_id"
    ],
)
mcp_tool.set_approval_mode("never")



toolset = ToolSet()
toolset.add(file_search)
toolset.add(functions)
toolset.add(mcp_tool)





# Enable automatic function calling 
project_client.agents.enable_auto_function_calls(toolset)


agent = project_client.agents.create_agent(
    model="gpt-4o",
    name="my-agent",
    instructions=open("instructions.txt").read(),
    top_p=0.7,
    temperature=0.7,
    toolset=toolset  # Add the toolset to the agent
)
print(f"Created agent, ID: {agent.id}")

thread = project_client.agents.threads.create()
print(f"Created thread, ID: {thread.id}")

while True:

    # Get the user input
    user_input = input("You: ")

    # Break out of the loop
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add a message to the thread
    message = project_client.agents.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER, 
        content=user_input
    )

    # Create and process an agent run
    run = project_client.agents.runs.create(
        thread_id=thread.id, 
        agent_id=agent.id, 
        tool_resources=mcp_tool.resources
    )

    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(0.1)
        run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)   

    messages = project_client.agents.messages.list(thread_id=thread.id)  
    first_message = next(iter(messages), None) 
    if first_message: 
        print(next((item["text"]["value"] for item in first_message.content if item.get("type") == "text"), ""))


project_client.agents.delete_agent(agent.id)
print("Deleted agent")


