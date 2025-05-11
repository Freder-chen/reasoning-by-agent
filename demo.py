import json
import asyncio
from typing import Literal
from pydantic import BaseModel

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    set_tracing_disabled,
    handoff,
    ItemHelpers,
    HandoffOutputItem,
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    ModelSettings,
)
from agents.model_settings import ModelSettings

# Disable tracing for cleaner output
set_tracing_disabled(True)

# System prompts
THINK_SYSTEM_PROMPT = """
Analyze the user's question and explore solutions. Reference prior messages for consistency. Separate solvable components from unresolved issues. Outline key steps briefly.
""".strip()

CRITIC_SYSTEM_PROMPT = """
Evaluate the answer's adequacy. Assign 'pass' or 'needs_improvement'. Provide specific actionable feedback if improvements are needed.

Output format:
{
  "score": "pass"|"needs_improvement",  // Choose one
  "feedback": "..."  // Specific, actionable feedback
}
""".strip()

ANSWER_SYSTEM_PROMPT = """
Synthesize tool responses and prior discussion into a clear, user-friendly answer. Address the original question directly, simplifying technical details without losing critical solutions.
""".strip()

PLAN_SYSTEM_PROMPT = """
Coordinate problem-solving by invoking one tool at a time as a helpful assistant.
""".strip()


# Define Pydantic models for structured data
class ChatPlan(BaseModel):
    task_description: str

class EvaluationFeedback(BaseModel):
    score: Literal["pass", "needs_improvement"]
    feedback: str


async def on_handoff(ctx: RunContextWrapper[None], input_data: ChatPlan):
    pass


class ConversationAgents:
    """Container class for all agents used in the conversation."""

    def __init__(self, model_name, base_url, api_key=None):
        self.vllm_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._initialize_agents(model_name)
    
    def _initialize_agents(self, model_name):
        model_config = {
            "model": model_name,
            "openai_client": self.vllm_client,
        }

        self.think_agent = Agent(
            name="think agent",
            model=OpenAIChatCompletionsModel(**model_config),
            instructions=THINK_SYSTEM_PROMPT,
        )
        
        self.answer_agent = Agent(
            name="answer agent",
            model=OpenAIChatCompletionsModel(**model_config),
            instructions=ANSWER_SYSTEM_PROMPT,
        )
        
        self.critic_agent = Agent(
            name="critic agent",
            model=OpenAIChatCompletionsModel(**model_config),
            instructions=CRITIC_SYSTEM_PROMPT,
            output_type=EvaluationFeedback,
        )
        
        self.chat_agent = Agent(
            name="plan agent",
            model=OpenAIChatCompletionsModel(**model_config),
            instructions=PLAN_SYSTEM_PROMPT,
            model_settings=ModelSettings(tool_choice="required"),
            handoffs=[
                handoff(
                    agent=self.think_agent,
                    on_handoff=on_handoff,
                    input_type=ChatPlan,
                ),
                handoff(
                    agent=self.answer_agent,
                    on_handoff=on_handoff,
                    input_type=ChatPlan,
                ),
                handoff(
                    agent=self.critic_agent,
                    on_handoff=on_handoff,
                    input_type=ChatPlan,
                ),
            ],
        )


class ConversationHandler:
    """Handles the conversation flow between agents."""
    
    def __init__(self, agents: ConversationAgents):
        self.agents = agents

    async def async_chat(self, initial_input: str, max_round=10) -> str:
        """Run the conversation loop until a satisfactory answer is reached."""
        # init message
        if isinstance(initial_input, str):
            initial_input = [{'role': 'user', 'content': initial_input}]

        run_input = initial_input        
        for i in range(max_round):
            # Run the chat agent to process the input
            result = await Runner.run(self.agents.chat_agent, input=run_input)

            # Log new items
            print('=' * 100)
            print(f"==> Step {i + 1}")
            self._log_new_items(result.new_items)

            # Check answer agent
            if result.last_agent is self.agents.answer_agent:
                # return result.final_output
                # Evaluate the final output using the critic agent
                evaluation_input = initial_input + [
                    {"content": f"{result.final_output}", "role": "assistant"},
                    {"content": f"Call critic agent and judge the answer", "role": "user"},
                ]
                status, feedback = await self._evaluate_output(self.agents.chat_agent, evaluation_input)
                if status == "pass":
                    return result.final_output
                elif status == "needs_improvement":
                    run_input = result.to_input_list() + [
                        {"content": f"Feedback: {feedback}", "role": "user"}
                    ]
                else:
                    raise NotImplementedError(f"The evaluation status `{status}` is not implemented.")

            else:
                run_input = result.to_input_list()
        
        return None
    
    def _log_new_items(self, new_items):
        """Parse the agent result and return appropriate next steps."""
        for new_item in new_items:
            agent_name = new_item.agent.name
            
            if isinstance(new_item, MessageOutputItem):
                output = ItemHelpers.text_message_output(new_item)
                print(f"=> {agent_name}: {output}")
            
            elif isinstance(new_item, HandoffOutputItem):
                print(f"=> Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}")
            
            elif isinstance(new_item, ToolCallItem):
                print(f"=> {agent_name}: Calling a tool")
            
            elif isinstance(new_item, ToolCallOutputItem):
                print(f"=> {agent_name}: Tool call output: {new_item.output}")
            
            else:
                new_input_item = new_item.to_input_item()
                arguments = json.loads(new_input_item['arguments'])
                print(f"=> {agent_name} call {new_input_item['name']}: {arguments}")
    
    async def _evaluate_output(self, agent, input) -> bool:
        """Evaluate the output with critic agent and determine if we should continue."""
        result = await Runner.run(
            agent,
            input=input
        )
        print('=' * 100)
        print('==> Critic answer:')
        self._log_new_items(result.new_items)

        evaluation = result.final_output
        status = evaluation.score
        feedback = evaluation.feedback
        return status, feedback


async def main():
    # Example usage
    user_question = "计算100！"
    # user_question = "如何高效计算100！"
    # user_question = "who are you?"
    # user_question = "What is the integer value of $x$ in the arithmetic sequence $3^2, x, 3^4$?"
    
    agents = ConversationAgents(
        model_name="qwen_instruct",
        base_url="http://0.0.0.0:8009/v1",
        api_key="EMPTY",
    )
    handler = ConversationHandler(agents)
    
    final_output = await handler.async_chat(user_question)
    print('=' * 50 + " Final Answer " + '=' * 50)
    print(final_output)


if __name__ == "__main__":
    asyncio.run(main())