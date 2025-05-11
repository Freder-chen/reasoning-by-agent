# Reasoning by Agent: `<think>` Token Is Unnecessary

随着Deepseek、Qwen3等模型陆续开源，大推理模型(Large Reasoning Models)正逐渐成为语言模型的新标准。其中，Deepseek-R1引入的`<think>`标签，试图通过新增Token显式标记推理过程，也被主流开源模型及框架迅速采纳和兼容。但实际上，我们并不需要`<think>`标签。

在观察o3、o4-mini的模型表现后，我更倾向于将它们视为Agent-as-a-Chatbot，或者更准确地说，是一种Reasoning-by-Agent的范式。它的核心思路是：通过调度器，将回答、推理、代码执行、搜索、图像识别等能力自然地整合为一个统一的智能体。在这种架构中，推理并不是通过Special Token显式触发，而是作为任务需求由系统自动调用。

相比之下，像Qwen3那样依赖`<think>\n</think>`来“隐藏“推理过程，其实是一种人为的、不自然的解决方案。与其依赖`<think>`标签去“提示模型开始思考”，我们更应该从系统架构层面去支持推理能力。这意味着，不再把“推理”当成语言模板，而是当成一种明确的能力模块，由调度器（Agent Controller）在合适的上下文中主动调用。

![image.png](asserts/image.png)

---

这种方式本质上是让LLM通过Agent框架给他自问自答的能力。我在[openai-agents-python](https://github.com/openai/openai-agents-python)找到了比较方便的实现。下面基于Qwen2.5-32B-Instruct和openai-agents-python实现一个示例，让大家直观理解这种做法：

```python
think_agent = Agent(
    name="think agent",
    model=OpenAIChatCompletionsModel(**model_config),
    instructions=THINK_SYSTEM_PROMPT,
)
answer_agent = Agent(
    name="answer agent",
    model=OpenAIChatCompletionsModel(**model_config),
    instructions=ANSWER_SYSTEM_PROMPT,
)
critic_agent = Agent(
    name="critic agent",
    model=OpenAIChatCompletionsModel(**model_config),
    instructions=CRITIC_SYSTEM_PROMPT,
    output_type=EvaluationFeedback,
)
chat_agent = Agent(
  name="plan agent",
  model=OpenAIChatCompletionsModel(**model_config),
  instructions=PLAN_SYSTEM_PROMPT,
  handoffs=[think_agent, answer_agent, critic_agent],
  model_settings=ModelSettings(tool_choice="required"),
)

user_question = "如何高效计算100！"
run_input = user_question
while True:
    # Run the chat agent to process the input
    result = await Runner.run(chat_agent, input=run_input)
    if result.last_agent is answer_agent:
        return result.final_output
    else:
        run_input = result.to_input_list()
```

```
Question：计算100！
====================================================================================================
==> Step 1
=> plan agent call transfer_to_think_agent: {'task_description': '计算100！需要一个能够处理大数运算的工具或方法。我们可以转移给能够执行这种复杂数学计算的代理。'}
=> Handed off from plan agent to think agent
=> think agent: 计算 \(100!\)（100的阶乘）意味着要计算从1乘到100的所有整数的乘积。这是一个非常大的数字，通常超出了普通计算器的处理能力。但是，可以使用编程语言或者专门的数学软件来完成这个计算。

这里有一个简单的Python代码示例来计算100!：

\`\`\`python
import math
print(math.factorial(100))
\`\`\`

这段代码会输出100的阶乘的结果。如果你没有编程环境，也可以使用在线的计算器或数学工具来得到结果。

如果你需要具体的数值，我可以通过其他方式帮你计算出来。你希望继续这个步骤吗？
====================================================================================================
==> Step 2
=> plan agent call transfer_to_answer_agent: {'task_description': '使用适当的工具或资源来计算100的阶乘，并提供结果。'}
=> Handed off from plan agent to answer agent
=> answer agent: \(100!\) 的值是一个非常大的数，具体为：

\[ 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000 \]

这是一个大约有158位的数字。由于这个数字非常庞大，日常生活中很少会直接用到这么大的数，但了解如何计算和表示它对于学习数学和计算机科学很有帮助。
====================================================================================================
==> Critic answer:
=> critic agent: {
  "score": "needs_improvement",
  "feedback": "The response provided the exact value of 100!, but it did not explain how to calculate factorials or provide context on why someone might need to know this. It would be beneficial to include a brief explanation of what a factorial is and perhaps a simple example or method for calculating smaller factorials."
}
====================================================================================================
==> Step 3
=> plan agent call transfer_to_answer_agent: {'task_description': '请提供有关阶乘的简要说明，包括定义、如何计算以及为什么在某些情况下需要知道这一点。此外，请给出一个小一点的阶乘计算作为例子。'}
=> Handed off from plan agent to answer agent
=> answer agent: 阶乘是一个数学概念，用于表示所有小于及等于该数的正整数的乘积。记作 \(n!\)，其中 \(n\) 是一个非负整数。例如，\(5!\) 表示 \(5 \times 4 \times 3 \times 2 \times 1 = 120\)。

阶乘常用于组合数学中，比如计算排列组合的数量。在概率论、统计学、计算机科学等领域中也会经常遇到阶乘的应用。例如，在确定一组项目有多少种不同的排序方式时，阶乘就显得非常重要。

回到 \(100!\) 的计算，由于其结果非常巨大，直接计算和展示这个数字主要用于理论研究或特定的计算任务。为了更好地理解阶乘的概念，我们来看一个更小的例子：计算 \(4!\)。

\[
4! = 4 \times 3 \times 2 \times 1 = 24
\]

这意味着四个不同物品的不同排列方式共有24种。

所以，\(100!\) 的确切值是：

\[ 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000 \]

这是一个非常大的数，展示了阶乘增长速度之快。希望这些信息能帮助你更好地理解和应用阶乘的概念。
====================================================================================================
==> Critic answer:
=> critic agent: {
  "score": "needs_improvement",
  "feedback": "The response includes unnecessary details about factorial usage in mathematics and an example of a smaller factorial calculation which was not asked for. The main issue is that while it provides the value of 100!, the explanation could be more concise and directly address the computation of 100! without additional context that wasn't requested. It should focus solely on calculating and explaining the process of computing 100!. Additionally, providing a step-by-step method or mentioning computational tools that can handle such large numbers would improve the answer."
}
====================================================================================================
==> Step 4
=> plan agent call transfer_to_answer_agent: {'task_description': '请专注于解释如何计算100!，并提供计算步骤或提及可以处理如此大数字的计算工具。'}
=> Handed off from plan agent to answer agent
=> answer agent: 计算 \(100!\) 涉及将从1到100的所有整数相乘。由于这个结果是一个非常大的数，手动计算是不现实的，通常需要借助计算机程序或专业的数学软件来完成。

### 计算步骤：
1. **定义阶乘**：\( n! = n \times (n-1) \times (n-2) \times ... \times 1 \)
2. **应用公式**：对于 \(100!\)，就是 \(100 \times 99 \times 98 \times ... \times 1\)。
3. **使用工具计算**：由于这个数字非常大，建议使用编程语言或数学软件进行计算。

### 使用Python计算 \(100!\)：
你可以使用Python内置的 `math` 库中的 `factorial` 函数来计算阶乘。下面是一个简单的Python代码示例：

\`\`\`python
import math
result = math.factorial(100)
print(result)
\`\`\`

运行上述代码将会输出 \(100!\) 的值：

\[ 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000 \]

### 使用其他工具：
除了Python，你还可以使用其他编程语言如Java、C++等，或者使用像WolframAlpha这样的在线计算器来计算阶乘。

通过这些工具，你可以轻松地处理和显示如此巨大的数字。希望这能帮助你理解如何计算 \(100!\)。
====================================================================================================
==> Critic answer:
=> critic agent: {
  "score": "pass",
  "feedback": "The response provided a detailed explanation on how to calculate 100! including a step-by-step breakdown, an example using Python programming, and suggestions for other tools. It was comprehensive and addressed the query effectively."
}
================================================== Final Answer ==================================================
计算 \(100!\) 涉及将从1到100的所有整数相乘。由于这个结果是一个非常大的数，手动计算是不现实的，通常需要借助计算机程序或专业的数学软件来完成。

### 计算步骤：
1. **定义阶乘**：\( n! = n \times (n-1) \times (n-2) \times ... \times 1 \)
2. **应用公式**：对于 \(100!\)，就是 \(100 \times 99 \times 98 \times ... \times 1\)。
3. **使用工具计算**：由于这个数字非常大，建议使用编程语言或数学软件进行计算。

### 使用Python计算 \(100!\)：
你可以使用Python内置的 `math` 库中的 `factorial` 函数来计算阶乘。下面是一个简单的Python代码示例：

\`\`\`python
import math
result = math.factorial(100)
print(result)
\`\`\`

运行上述代码将会输出 \(100!\) 的值：

\[ 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000 \]

### 使用其他工具：
除了Python，你还可以使用其他编程语言如Java、C++等，或者使用像WolframAlpha这样的在线计算器来计算阶乘。

通过这些工具，你可以轻松地处理和显示如此巨大的数字。希望这能帮助你理解如何计算 \(100!\)。
```

---

如果大佬们觉得这个方法有效，希望出训练框架和模型。没钱训QAQ

代码：[github](https://github.com/Freder-chen/reasoning-by-agent)